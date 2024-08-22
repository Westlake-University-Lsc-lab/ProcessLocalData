import numpy as np
from itertools import islice
from typing import Literal, List
from collections import namedtuple
from typing import Union

__all__ = [
    "DAWDemoWaveParser",
]

def _bytes_to_int(
        data: bytes, 
        bit: "int | None"=None, 
        start: int = 0, 
        byteorder: Literal['little', 'big'] = 'little'
    ) -> int:
    """
        Return given bits from binary `bytes`.

        Parameters
        ----------
        data : bytes
            Bytes that will be converted to int.
        start : int
            The first bit to convert. Default = 0.
        bit : int
            Number of bits for truncating. For example, with `bit=5`, binary 
            number `0b10111010` will be converted to `0b11010`. Default `None`
            and all bits will be preserved.
        byteorder : {'little', 'big'}
            The byte order used to represent the integer. Default `little`.

        Example
        -------
        >>> data = b'\\x10\\x0a'
                    # from little: 00001010 00010000
                    # from big:    00001000 00001010
        >>> _bytes_to_int(data, 10, byteorder = 'little')
        528             # 0b000010(10 00010000) => 0b10 00010000 == 528
        >>> _bytes_to_int(data, 10, byteorder = 'big')
        10              # 0b000010(00 00001010) => 0b00 00001010 == 10
        >>> _bytes_to_int(data, 10, 1, byteorder = 'little')
        264             # 0b0001(010 0001000)0 => 0b01 00001000 == 264
    """
    full_num = int.from_bytes(bytes=data, byteorder=byteorder) >> start
    return full_num if bit is None else (full_num & ((1 << bit) - 1))



def _one_loc(num: int) -> List[int]:
    """
        Return a list of the location of ones in the binary representation of `num`.

        Parameters
        ----------
        num : int
            The number to get non-zero bits.

        Example
        --------
        ```python
        >>> data = 19
        >>> bin(data)
        '0b10011'
        >>> _one_loc(data)
        [0, 1, 4]
        ```
    """
    index_list = []
    bit = 0
    while num != 0:
        if num & 1:
            index_list.append(bit)
        bit += 1
        num >>= 1
    return index_list

_Wave = namedtuple('Wave', ['Channel', 'Timestamp', 'Trunc', 'Baseline', 'Waveform'])

class DAWDemoWaveParser:
    """
    An iterable class, which will yield a tuple `(channel, time, signal)` in binary file(s) when iterated.
    """
    
    def __init__(self, raw: "str | list[str]") -> None:
        self._raw = [raw] if (type(raw) is str) else raw

    def _generator(self):
        for file in self._raw:
            with open(file , mode='rb') as f:

                # Read All Events
                # ---------------
                ## One line contains 4 Bytes. That means total bytes = lines << 2.
                ## Reader header for a single event. That contains 4 lines.
                ## Read file until file pointer reaches EOF.
                while event_header := f.read(4 << 2):

                    # Get active channels
                    # --------------------
                    ## Masks 0~7: In Byte 0 of line 1
                    ## Masks 8~15: In Byte 3 of line 3
                    channels = _one_loc(event_header[4] + (event_header[11] << 8))

                    for ch in channels:
                        ## Read header of this channels
                        ch_header = f.read(3 << 2)

                        # Get event size of this channel.
                        # ----------------------------
                        ## Line count of the header is 3, the rest is the signal.
                        ch_size = _bytes_to_int(ch_header[:3], 22)
                        sig_size = (ch_size - 3) << 2
                        #time_stamp = _bytes_to_int(ch_header[4:10],6)
                        time_stamp = _bytes_to_int(ch_header[4:10])

                        # Get Truncation flag and baseline (given by DAW software)
                        trunc = bool((ch_header[3] >> 6) & 1)
                        #baseline = _bytes_to_int(ch_header[10:12],2)
                        baseline = _bytes_to_int(ch_header[10:12])

                        # Get channel raw signal and process
                        # -----------------------------------
                        ## Every 2 Bytes represent an ADC signal
                        raw_sig = f.read(sig_size)      # binary signal list
                        sig = np.frombuffer(raw_sig, dtype=np.int16)

                        ## yeild channel data
                        yield _Wave(ch, time_stamp, trunc, baseline, sig)
    
    def binary_generator(self):
        """
        Read out the binary data corresponding to each waveform.
        """
        for file in self._raw:
            with open(file , mode='rb') as f:
                
                while event_header := f.read(4 << 2):
                    event_line_count = _bytes_to_int(event_header[:4], 28)
                    event_body_size = (event_line_count - 4) << 2
                    event_body = f.read(event_body_size)
                    yield event_header + event_body

    def __iter__(self):
        return self._generator()
    
    def __getitem__(self, key):
        # If key is an integer, then return a single waveform
        # loop from file start. 
        if isinstance(key, int):
            empty_err = np.array([], dtype='i2')
            return next(islice(self, key, key+1), empty_err)
        
        # if key is a slice([::]), then return a islice object
        elif isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            return islice(self, start, stop, step)

    def __repr__(self) -> str:
        file_counter = len(self._raw)
        return f"< DAW Demo binary file loader | From {file_counter} file{'s' if file_counter > 1 else ''}. >"
