// Package twiml builds the small TwiML documents the scheduler returns to Twilio.
//
// We use encoding/xml (rather than string concatenation) so attribute values are
// correctly escaped — a caller-supplied <Parameter> value containing & or " must
// not be able to corrupt the document.
package twiml

import "encoding/xml"

type response struct {
	XMLName xml.Name `xml:"Response"`
	Connect *connect `xml:"Connect,omitempty"`
	Reject  *reject  `xml:"Reject,omitempty"`
	Say     string   `xml:"Say,omitempty"`
}

type connect struct {
	Stream stream `xml:"Stream"`
}

type stream struct {
	URL        string      `xml:"url,attr"`
	Parameters []parameter `xml:"Parameter"`
}

type parameter struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

type reject struct {
	Reason string `xml:"reason,attr,omitempty"`
}

// ConnectStream returns TwiML that opens a BIDIRECTIONAL media stream to wsURL.
//
// <Connect><Stream> (not <Start><Stream>) is required for a voice bot: it is
// bidirectional (we can play audio back) and it BLOCKS the call for the stream's
// lifetime. The params become <Parameter> children and surface to the Python side
// inside Twilio's `start` event under start.customParameters — this is how call
// context (callSid, tenant, engine) rides along to the media socket.
func ConnectStream(wsURL string, params map[string]string) ([]byte, error) {
	ps := make([]parameter, 0, len(params))
	for k, v := range params {
		ps = append(ps, parameter{Name: k, Value: v})
	}
	doc := response{Connect: &connect{Stream: stream{URL: wsURL, Parameters: ps}}}
	return marshal(doc)
}

// Reject returns TwiML that declines the call — used when the scheduler is at
// capacity (fair-share backpressure) or the backend refuses the call.
func Reject(reason string) ([]byte, error) {
	return marshal(response{Reject: &reject{Reason: reason}})
}

// Busy returns a friendlier "all agents busy" message instead of a hard reject.
func Busy(message string) ([]byte, error) {
	return marshal(response{Say: message})
}

func marshal(doc response) ([]byte, error) {
	body, err := xml.Marshal(doc)
	if err != nil {
		return nil, err
	}
	return append([]byte(xml.Header), body...), nil
}
