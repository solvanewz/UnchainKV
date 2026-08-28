import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import unchain_kv.splitzip_cuda as splitzip_cuda


class SplitzipCudaTest(unittest.TestCase):
    def test_top16_layer_bound_follows_codebook_count(self):
        layers = len(splitzip_cuda._TOP16_CODEBOOKS) // 32
        self.assertIsNone(splitzip_cuda._top16_tables(layers, "cuda", object()))

    def test_top16_wrappers_pass_static_tables_to_cuda(self):
        encoder = getattr(splitzip_cuda, "encode_top16", None)
        decoder = getattr(splitzip_cuda, "decode_top16", None)
        self.assertIsNotNone(encoder)
        self.assertIsNotNone(decoder)
        if encoder is None or decoder is None:
            return

        encode_calls = []
        decode_calls = []

        class FakeLib:
            def kvq_splitzip_bf16_encode_top16(self, *args):
                encode_calls.append(args)
                return 13

            def kvq_splitzip_bf16_decode_top16(self, *args):
                decode_calls.append(args)
                return 16

        class FakeTensor:
            def __init__(self, ptr, shape, stride, dtype, device):
                self._ptr = ptr
                self.shape = shape
                self._stride = stride
                self.dtype = dtype
                self.device = device

            def data_ptr(self):
                return self._ptr

            def numel(self):
                result = 1
                for size in self.shape:
                    result *= size
                return result

            def dim(self):
                return len(self.shape)

            def stride(self, dim):
                return self._stride[dim]

            def is_contiguous(self):
                return True

        device = SimpleNamespace(type="cuda")
        uint8 = object()
        bfloat16 = object()
        source = FakeTensor(11, (8,), (1,), bfloat16, device)
        encoded = FakeTensor(22, (32,), (1,), uint8, device)
        target = FakeTensor(33, (2, 8, 2), (16, 2, 1), bfloat16, device)
        tables = FakeTensor(44, (544,), (1,), uint8, device)
        fake_torch = SimpleNamespace(
            uint8=uint8,
            bfloat16=bfloat16,
            cuda=SimpleNamespace(
                current_stream=lambda _device: SimpleNamespace(cuda_stream=99)
            ),
        )
        old_lib = splitzip_cuda._LIB
        splitzip_cuda._LIB = FakeLib()
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                splitzip_cuda, "_top16_tables", return_value=tables, create=True
            ):
                encoded_bytes = encoder(source, encoded, 7)
                copied = decoder(encoded, target, [3, 4], 16, 7)
        finally:
            splitzip_cuda._LIB = old_lib

        self.assertEqual(encoded_bytes, 13)
        self.assertEqual(copied, 16)
        self.assertEqual(encode_calls, [(11, 22, 8, 32, 44, 99)])
        self.assertEqual(
            decode_calls, [(22, 32, 33, 8, 2, 2, 16, 2, 3, 44, 99)]
        )

    def test_fixed6_cuda_round_trip_writes_noncontiguous_blocks(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not splitzip_cuda.available():
            self.skipTest(splitzip_cuda.load_error() or "splitzip CUDA unavailable")

        count = 2 * 128 * 2048
        index = torch.arange(count, dtype=torch.int32, device="cuda")
        palette = torch.tensor([0x3F, 0x40, 0xBF, 0xC0], device="cuda")
        high = palette[torch.remainder(index, 4)]
        words = torch.bitwise_or(
            torch.bitwise_left_shift(high, 8), torch.remainder(index, 251)
        ).to(torch.uint16)
        source = words.view(torch.bfloat16).reshape(2, 128, 2048)
        raw_bytes = source.numel() * source.element_size()
        encoded_buffer = torch.empty(raw_bytes, dtype=torch.uint8, device="cuda")

        encoded_bytes = splitzip_cuda.encode_bf16(source, encoded_buffer)
        self.assertIsNotNone(encoded_bytes)
        target = torch.zeros((2, 256, 2048), dtype=torch.bfloat16, device="cuda")
        block_ids = list(range(64, 128)) + list(range(192, 256))
        copied = splitzip_cuda.decode_fixed6(
            encoded_buffer[:encoded_bytes], target, block_ids, raw_bytes
        )
        torch.cuda.synchronize()

        self.assertEqual(copied, raw_bytes)
        self.assertTrue(
            torch.equal(target[:, block_ids].view(torch.uint16), source.view(torch.uint16))
        )

    def test_top16_cuda_round_trip_preserves_escapes_for_small_and_fragmented(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not splitzip_cuda.available():
            self.skipTest(splitzip_cuda.load_error() or "splitzip CUDA unavailable")

        for block_count, target_blocks, block_ids in (
            (1, 4, [3]),
            (128, 256, list(range(0, 256, 2))),
        ):
            with self.subTest(block_count=block_count):
                count = 2 * block_count * 2048
                index = torch.arange(count, dtype=torch.int32, device="cuda")
                codebooks = torch.tensor(
                    list(splitzip_cuda._TOP16_CODEBOOKS[:32]), device="cuda"
                ).reshape(2, 16)
                plane = torch.div(index, count // 2, rounding_mode="floor")
                exponent = codebooks[plane.long(), torch.remainder(index, 16).long()]
                exponent[::400] = 100
                words = torch.bitwise_or(
                    torch.bitwise_left_shift(torch.remainder(index, 2), 15),
                    torch.bitwise_or(
                        torch.bitwise_left_shift(exponent.to(torch.int32), 7),
                        torch.remainder(index, 128),
                    ),
                ).to(torch.uint16)
                source = words.view(torch.bfloat16).reshape(2, block_count, 2048)
                raw_bytes = source.numel() * source.element_size()
                encoded_buffer = torch.empty(
                    raw_bytes, dtype=torch.uint8, device="cuda"
                )

                encoded_bytes = splitzip_cuda.encode_top16(source, encoded_buffer, 0)
                self.assertIsNotNone(encoded_bytes)
                target = torch.zeros(
                    (2, target_blocks, 2048), dtype=torch.bfloat16, device="cuda"
                )
                copied = splitzip_cuda.decode_top16(
                    encoded_buffer[:encoded_bytes],
                    target,
                    block_ids,
                    raw_bytes,
                    0,
                )
                torch.cuda.synchronize()

                self.assertGreaterEqual(raw_bytes / encoded_bytes, 1.30)
                self.assertEqual(int(encoded_buffer[0].item()), 5)
                self.assertEqual(copied, raw_bytes)
                self.assertTrue(
                    torch.equal(
                        target[:, block_ids].view(torch.uint16),
                        source.view(torch.uint16),
                    )
                )

    def test_decode_fixed6_passes_contiguous_native_layout_to_cuda(self):
        decoder = getattr(splitzip_cuda, "decode_fixed6", None)
        self.assertIsNotNone(decoder)
        if decoder is None:
            return

        calls = []

        class FakeLib:
            def kvq_splitzip_bf16_decode_fixed6(self, *args):
                calls.append(args)
                return 16

        class FakeTensor:
            def __init__(self, ptr, shape, stride, dtype, device):
                self._ptr = ptr
                self.shape = shape
                self._stride = stride
                self.dtype = dtype
                self.device = device

            def data_ptr(self):
                return self._ptr

            def numel(self):
                result = 1
                for size in self.shape:
                    result *= size
                return result

            def dim(self):
                return len(self.shape)

            def stride(self, dim):
                return self._stride[dim]

            def is_contiguous(self):
                return True

        device = SimpleNamespace(type="cuda")
        uint8 = object()
        bfloat16 = object()
        source = FakeTensor(11, (79,), (1,), uint8, device)
        target = FakeTensor(22, (2, 8, 2), (16, 2, 1), bfloat16, device)
        fake_torch = SimpleNamespace(
            uint8=uint8,
            bfloat16=bfloat16,
            cuda=SimpleNamespace(
                current_stream=lambda _device: SimpleNamespace(cuda_stream=99)
            ),
        )
        old_lib = splitzip_cuda._LIB
        splitzip_cuda._LIB = FakeLib()
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}):
                copied = decoder(source, target, [3, 4], 16)
        finally:
            splitzip_cuda._LIB = old_lib

        self.assertEqual(copied, 16)
        self.assertEqual(calls, [(11, 79, 22, 8, 2, 2, 16, 2, 3, 99)])

    def test_native_decoders_pass_noncontiguous_block_map_to_cuda(self):
        fixed_calls = []
        top16_calls = []

        class FakeLib:
            def kvq_splitzip_bf16_decode_fixed6(self, *_args):
                raise AssertionError("contiguous decoder used")

            def kvq_splitzip_bf16_decode_fixed6_blocks(self, *args):
                fixed_calls.append(args)
                return 16

            def kvq_splitzip_bf16_decode_top16(self, *_args):
                raise AssertionError("contiguous decoder used")

            def kvq_splitzip_bf16_decode_top16_blocks(self, *args):
                top16_calls.append(args)
                return 16

        class FakeTensor:
            def __init__(self, ptr, shape, stride, dtype, device):
                self._ptr = ptr
                self.shape = shape
                self._stride = stride
                self.dtype = dtype
                self.device = device

            def data_ptr(self):
                return self._ptr

            def numel(self):
                result = 1
                for size in self.shape:
                    result *= size
                return result

            def dim(self):
                return len(self.shape)

            def stride(self, dim):
                return self._stride[dim]

            def is_contiguous(self):
                return True

        class FakeBlockIds(FakeTensor):
            def __init__(self):
                super().__init__(55, (2,), (1,), int64, device)
                self.streams = []

            def record_stream(self, stream):
                self.streams.append(stream)

        device = SimpleNamespace(type="cuda")
        uint8 = object()
        bfloat16 = object()
        int64 = object()
        source = FakeTensor(11, (79,), (1,), uint8, device)
        target = FakeTensor(22, (2, 8, 2), (16, 2, 1), bfloat16, device)
        tables = FakeTensor(44, (544,), (1,), uint8, device)
        block_map = FakeBlockIds()
        current = SimpleNamespace(cuda_stream=99)
        fake_torch = SimpleNamespace(
            uint8=uint8,
            bfloat16=bfloat16,
            int64=int64,
            tensor=lambda *_args, **_kwargs: block_map,
            cuda=SimpleNamespace(current_stream=lambda _device: current),
        )
        old_lib = splitzip_cuda._LIB
        splitzip_cuda._LIB = FakeLib()
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                splitzip_cuda, "_top16_tables", return_value=tables, create=True
            ):
                fixed = splitzip_cuda.decode_fixed6(source, target, [3, 5], 16)
                top16 = splitzip_cuda.decode_top16(source, target, [3, 5], 16, 7)
        finally:
            splitzip_cuda._LIB = old_lib

        self.assertEqual((fixed, top16), (16, 16))
        self.assertEqual(
            fixed_calls, [(11, 79, 22, 8, 2, 2, 16, 2, 3, 55, 99)]
        )
        self.assertEqual(
            top16_calls, [(11, 79, 22, 8, 2, 2, 16, 2, 3, 55, 44, 99)]
        )
        self.assertEqual(block_map.streams, [current, current])


if __name__ == "__main__":
    unittest.main()
