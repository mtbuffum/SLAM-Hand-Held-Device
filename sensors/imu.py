import json
import serial


class IMUReader:
    def __init__(self, port: str, baud: int = 115200):
        # timeout=0 makes reads non-blocking
        self.ser = serial.Serial(port, baudrate=baud, timeout=0)
        self.buf = bytearray()
        self.latest = None

    def read_latest(self):
        """
        Reads whatever bytes are available and returns the most recent parsed JSON sample dict.
        Returns None if nothing new yet.
        """
        chunk = self.ser.read(4096)
        if chunk:
            self.buf.extend(chunk)

            while b"\n" in self.buf:
                line, _, rest = self.buf.partition(b"\n")
                self.buf = bytearray(rest)

                line = line.strip()
                if not line:
                    continue

                try:
                    d = json.loads(line.decode("utf-8"))
                    # sanity check required fields
                    if all(k in d for k in ("t_us", "gx", "gy", "gz", "ax", "ay", "az")):
                        self.latest = d
                except Exception:
                    # ignore malformed lines
                    pass

        return self.latest
