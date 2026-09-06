# genesys-memory is now papez

The package was renamed. Install `papez` and import `papez`:

```bash
pip install papez
```

This final `genesys-memory` release depends on `papez` and keeps every
`import genesys_memory...` statement working, with a `DeprecationWarning`, so
existing code keeps running while you update it. It will not receive fixes.
