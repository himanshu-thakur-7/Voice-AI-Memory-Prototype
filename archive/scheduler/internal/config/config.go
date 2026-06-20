// Package config loads the scheduler's settings from the environment.
//
// Go onboarding note: there's no global singleton here. We load once in main()
// and pass the struct down explicitly (dependency injection by hand). This keeps
// every component testable — a test can construct its own Config without touching
// process state.
package config

import (
	"os"
	"strconv"
	"time"
)

// Config is the fully-resolved scheduler configuration.
type Config struct {
	HTTPListenAddr     string        // where the Twilio webhook listens, e.g. ":8080"
	MediaWSURL         string        // wss:// URL we embed in TwiML for Twilio to stream to
	GRPCBackendAddr    string        // host:port of the Python Orchestrator gRPC server
	TwilioAuthToken    string        // used to validate X-Twilio-Signature
	Engine             string        // "cascade" | "realtime" — forwarded to the backend per call
	MaxConcurrentCalls int           // global fair-share ceiling
	AdmitTimeout       time.Duration // how long a webhook will wait for an admission slot
	ShutdownGrace      time.Duration // drain window on SIGTERM
}

// Load reads configuration from the environment, applying sane defaults so the
// service runs locally with an empty .env.
func Load() Config {
	return Config{
		HTTPListenAddr:     getenv("HTTP_LISTEN_ADDR", ":8080"),
		MediaWSURL:         getenv("MEDIA_WS_URL", "wss://localhost:8000/media"),
		GRPCBackendAddr:    getenv("GRPC_BACKEND_ADDR", "localhost:50051"),
		TwilioAuthToken:    getenv("TWILIO_AUTH_TOKEN", ""),
		Engine:             getenv("ENGINE", "cascade"),
		MaxConcurrentCalls: getenvInt("MAX_CONCURRENT_CALLS", 200),
		AdmitTimeout:       getenvDuration("ADMIT_TIMEOUT", 2*time.Second),
		ShutdownGrace:      getenvDuration("SHUTDOWN_GRACE", 25*time.Second),
	}
}

func getenv(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func getenvInt(key string, def int) int {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getenvDuration(key string, def time.Duration) time.Duration {
	if v, ok := os.LookupEnv(key); ok {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
