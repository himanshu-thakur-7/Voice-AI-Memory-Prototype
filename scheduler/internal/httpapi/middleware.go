package httpapi

import (
	"crypto/hmac"
	"crypto/sha1"
	"crypto/subtle"
	"encoding/base64"
	"log/slog"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

// validateTwilioSignature implements Twilio's request-signature scheme so we can
// trust that a /voice POST really came from Twilio before doing any work.
//
// Algorithm (form-encoded POST):
//  1. Start with the full request URL exactly as Twilio called it (scheme+host+path+query).
//  2. Append each POST param, sorted by key, as key immediately followed by value
//     (no separators), concatenated onto the URL.
//  3. HMAC-SHA1 the result using the account AuthToken as the key.
//  4. Base64-encode and compare (constant time) to the X-Twilio-Signature header.
//
// In production you'd typically use github.com/twilio/twilio-go's RequestValidator
// instead of hand-rolling this — but implementing it once is the clearest way to
// understand what that library actually checks, and keeps this service dependency-free.
func validateTwilioSignature(authToken, fullURL string, form url.Values, signature string) bool {
	keys := make([]string, 0, len(form))
	for k := range form {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	b.WriteString(fullURL)
	for _, k := range keys {
		// Twilio joins multi-valued params by concatenation in order.
		for _, v := range form[k] {
			b.WriteString(k)
			b.WriteString(v)
		}
	}

	mac := hmac.New(sha1.New, []byte(authToken))
	mac.Write([]byte(b.String()))
	expected := base64.StdEncoding.EncodeToString(mac.Sum(nil))

	// Constant-time compare to avoid leaking validity via timing.
	return subtle.ConstantTimeCompare([]byte(expected), []byte(signature)) == 1
}

// signatureGuard wraps a handler, rejecting requests whose X-Twilio-Signature does
// not validate. If authToken is empty (local dev), it logs once and lets traffic
// through so the prototype runs without Twilio credentials.
func signatureGuard(authToken, publicBaseURL string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if authToken == "" {
			slog.Warn("TWILIO_AUTH_TOKEN unset — skipping signature validation (DEV ONLY)")
			next(w, r)
			return
		}
		if err := r.ParseForm(); err != nil {
			http.Error(w, "bad form", http.StatusBadRequest)
			return
		}
		// The signed URL is the PUBLIC URL Twilio dialed, which behind ngrok/load
		// balancers is not r.Host. We reconstruct it from the configured public base.
		fullURL := strings.TrimRight(publicBaseURL, "/") + r.URL.Path
		if r.URL.RawQuery != "" {
			fullURL += "?" + r.URL.RawQuery
		}
		sig := r.Header.Get("X-Twilio-Signature")
		if !validateTwilioSignature(authToken, fullURL, r.PostForm, sig) {
			slog.Warn("rejected request with invalid Twilio signature", "path", r.URL.Path)
			http.Error(w, "invalid signature", http.StatusForbidden)
			return
		}
		next(w, r)
	}
}
