module github.com/voiceai/scheduler

go 1.22

// NOTE: this service is intentionally STDLIB-ONLY so it builds and tests offline
// with zero codegen. Two production swap-ins are documented inline:
//   • Twilio signature validation → github.com/twilio/twilio-go (RequestValidator)
//   • the gRPC client stub        → the generated ./internal/pb stubs (`make proto`)
// When you add them, this `require` block is where they land.
