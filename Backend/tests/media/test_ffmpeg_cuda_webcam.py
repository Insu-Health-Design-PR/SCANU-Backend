from media.capture.ffmpeg_cuda_webcam import build_ffmpeg_cuda_argv, scaled_output_size


def test_scaled_output_size_4k_to_1920():
    assert scaled_output_size(3840, 2160, 1920) == (1920, 1080)


def test_scaled_output_size_keeps_small():
    assert scaled_output_size(1280, 720, 1920) == (1280, 720)


def test_build_ffmpeg_cuda_argv_has_cuvid_and_scale():
    argv = build_ffmpeg_cuda_argv(
        "/dev/video0",
        width=3840,
        height=2160,
        fps=60,
        out_width=1920,
        out_height=1080,
    )
    joined = " ".join(argv)
    assert "mjpeg_cuvid" in joined
    assert "scale_cuda=1920:1080" in joined
    assert "/dev/video0" in argv
    assert argv[-1] == "pipe:1"


def test_build_ffmpeg_cuda_argv_dual_full_res():
    argv = build_ffmpeg_cuda_argv(
        "/dev/video0",
        width=3840,
        height=2160,
        fps=60,
        out_width=1920,
        out_height=1080,
        full_pipe_fd=5,
        full_fps=30,
    )
    joined = " ".join(argv)
    assert "split=2" in joined
    assert "scale_cuda=1920:1080" in joined
    assert "fps=30" in joined
    assert "pipe:1" in argv
    assert "pipe:5" in argv
