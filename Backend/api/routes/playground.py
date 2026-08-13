"""Single-image inference playground routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.routes.context import RouterContext
from api.schemas.playground import PlaygroundInferBody

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui import playground as playground_infer  # noqa: E402


def build_playground_router(ctx: RouterContext) -> APIRouter:
    layer8_dir = ctx.layer8_dir
    router = APIRouter(tags=["playground"])

    @router.post("/api/playground/infer_image")
    def playground_infer_image(body: PlaygroundInferBody) -> dict:
        upload_path: Path | None = None
        try:
            if body.image_b64 and body.image_b64.strip():
                upload_path = playground_infer.save_uploaded_image_b64(body.image_b64.strip())
                source = upload_path
            elif body.use_sample:
                source = playground_infer.default_sample_image_path(layer8_dir)
            else:
                raise HTTPException(400, "provide image_b64 or set use_sample=true")
            if not source.is_file():
                raise HTTPException(404, f"image not found: {source}")
            return playground_infer.run_playground_infer(
                layer8_dir=layer8_dir,
                source=source,
                values=body.values,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if upload_path is not None and upload_path.is_file():
                try:
                    upload_path.unlink()
                except OSError:
                    pass

    @router.get("/api/playground/sample")
    def playground_sample_image() -> FileResponse:
        try:
            path = playground_infer.default_sample_image_path(layer8_dir)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(str(path), media_type="image/jpeg")

    return router
