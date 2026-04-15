"""Per-model loaders for the inemaimg registry.

Each loader exposes two classmethods:

    load() -> pipeline
        Instantiates the underlying diffusers pipeline, moves it to CUDA,
        and returns it ready to infer.

    generate(pipe, req, images) -> PIL.Image.Image
        Runs one generation using the validated GenerateRequest and the
        decoded PIL reference images (empty list for text-to-image models).

The server only talks to loaders through these two methods, so adding a
new model means dropping a new file in this package and registering it
in server.REGISTRY.
"""
