// Command scheduler is Module 1: the Twilio-facing webhook + Fair-Share orchestrator.
//
// Lifecycle:
//   1. Load config from the environment.
//   2. Start the Fair-Share scheduler's dispatcher goroutine.
//   3. Dial the Python backend (gRPC; stub by default).
//   4. Serve the webhook HTTP API.
//   5. On SIGINT/SIGTERM, stop accepting calls and drain in-flight work.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/voiceai/scheduler/internal/config"
	"github.com/voiceai/scheduler/internal/grpcclient"
	"github.com/voiceai/scheduler/internal/httpapi"
	"github.com/voiceai/scheduler/internal/scheduler"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})))

	cfg := config.Load()

	// rootCtx is canceled on the first SIGINT/SIGTERM. Everything hangs off it, so a
	// single signal cleanly unwinds the scheduler, the HTTP server, and the gRPC client.
	rootCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// (2) Fair-Share scheduler. Run() blocks, so it gets its own goroutine; it stops
	// when rootCtx is canceled.
	sched := scheduler.New(cfg.MaxConcurrentCalls)
	go sched.Run(rootCtx)

	// (3) Backend link (logging stub until `make proto` + dialGRPC swap).
	backend := grpcclient.New(cfg.GRPCBackendAddr)
	defer func() { _ = backend.Close() }()

	// (4) HTTP server.
	srv := &http.Server{
		Addr:              cfg.HTTPListenAddr,
		Handler:           httpapi.NewServer(cfg, sched, backend).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	// Run ListenAndServe in a goroutine so main can wait on the signal context.
	serveErr := make(chan error, 1)
	go func() {
		slog.Info("scheduler listening", "addr", cfg.HTTPListenAddr,
			"engine", cfg.Engine, "maxConcurrent", cfg.MaxConcurrentCalls)
		serveErr <- srv.ListenAndServe()
	}()

	// (5) Block until either the server dies or we get a shutdown signal.
	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server error", "err", err)
		}
	case <-rootCtx.Done():
		slog.Info("shutdown signal received; draining", "grace", cfg.ShutdownGrace)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			slog.Error("graceful shutdown failed", "err", err)
		}
	}
	slog.Info("scheduler stopped")
}
