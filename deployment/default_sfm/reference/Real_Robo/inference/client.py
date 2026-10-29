"""HTTP/NPZ client; no model or robot dependencies."""
import uuid
import requests

class HTTPPolicyClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        from .protocol import PROTOCOL
        self.session.headers["X-Policy-Protocol"] = PROTOCOL

    def reset(self, text: str, previous_episode=None):
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"text": text, "previous_episode": previous_episode},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self):
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        from .protocol import validate_transport
        result = resp.json()
        validate_transport(result)
        return result

    def infer(self, observation: dict):
        from .protocol import PROTOCOL, MEDIA_TYPE, pack_arrays, unpack_arrays
        request_id = uuid.uuid4().hex
        # No automatic retries: a stateful policy must not advance twice for
        # one observation after an ambiguous timeout.
        resp = self.session.post(
            f"{self.base_url}/query", data=pack_arrays(observation),
            headers={"Content-Type": MEDIA_TYPE, "Accept": MEDIA_TYPE,
                     "X-Request-Id": request_id}, timeout=(3.05, 60))
        resp.raise_for_status()
        if (resp.headers.get('X-Policy-Protocol') != PROTOCOL or
                resp.headers.get('X-Request-Id') != request_id or
                resp.headers.get('Content-Type', '').split(';')[0] != MEDIA_TYPE):
            raise ValueError('Policy response protocol, request id or content type does not match')
        result = unpack_arrays(resp.content)
        if set(result) != {'actions'}:
            raise ValueError('Policy response must contain only the actions array')
        return result

