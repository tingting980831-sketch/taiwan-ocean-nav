File "/mount/src/taiwan-ocean-nav/app.py", line 42, in <module>
    era5_lons, era5_lats, era5_u10, era5_v10, era5_ws, era5_swh = load_era5_ocean()
                                                                  ~~~~~~~~~~~~~~~^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 281, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 326, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/caching/cache_utils.py", line 385, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
File "/mount/src/taiwan-ocean-nav/app.py", line 26, in load_era5_ocean
    ds = xr.open_dataset(url, engine="zarr",
                         storage_options={"client_kwargs": {"trust_env": True}})
File "/home/adminuser/venv/lib/python3.13/site-packages/xarray/backends/api.py", line 607, in open_dataset
    backend_ds = backend.open_dataset(
        filename_or_obj,
    ...<2 lines>...
        **kwargs,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/xarray/backends/zarr.py", line 1683, in open_dataset
    store = ZarrStore.open_group(
        filename_or_obj,
    ...<10 lines>...
        cache_members=cache_members,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/xarray/backends/zarr.py", line 722, in open_group
    ) = _get_open_params(
        ~~~~~~~~~~~~~~~~^
        store=store,
        ^^^^^^^^^^^^
    ...<9 lines>...
        zarr_format=zarr_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/xarray/backends/zarr.py", line 1891, in _get_open_params
    zarr_root_group = zarr.open_consolidated(store, **open_kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/api/synchronous.py", line 238, in open_consolidated
    sync(async_api.open_consolidated(*args, use_consolidated=use_consolidated, **kwargs))
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/core/sync.py", line 159, in sync
    raise return_result
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/core/sync.py", line 119, in _runner
    return await coro
           ^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/api/asynchronous.py", line 415, in open_consolidated
    return await open_group(*args, use_consolidated=use_consolidated, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/api/asynchronous.py", line 866, in open_group
    return await AsyncGroup.open(
           ^^^^^^^^^^^^^^^^^^^^^^
        store_path, zarr_format=zarr_format, use_consolidated=use_consolidated
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/core/group.py", line 570, in open
    ) = await asyncio.gather(
        ^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/storage/_common.py", line 168, in get
    return await self.store.get(self.path, prototype=prototype, byte_range=byte_range)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/zarr/storage/_fsspec.py", line 289, in get
    value = prototype.buffer.from_bytes(await self.fs._cat_file(path))
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/fsspec/implementations/http.py", line 247, in _cat_file
    self._raise_not_found_for_status(r, url)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/fsspec/implementations/http.py", line 230, in _raise_not_found_for_status
    response.raise_for_status()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
File "/home/adminuser/venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py", line 636, in raise_for_status
    raise ClientResponseError(
    ...<5 lines>...
    )
