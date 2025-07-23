#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gsm7 → gsm9 変換
  ・正規化・量子化なし（dtype は元ファイルをそのまま踏襲）
  ・チャンク = (1,128,128)
  ・圧縮は blosc-LZ4
出力 : ./128_128/nc_gsm9
"""
import os, time
from glob import glob
from datetime import datetime
import numpy as np
from netCDF4 import Dataset

# ───────── 1. チャンクサイズ計算 ─────────
def chunksizes(var, dimdict):
    ch = []
    for d in var.dimensions:
        if d == 'time':
            ch.append(1)
        elif d in ('lat', 'lon'):
            ch.append(min(128, len(dimdict[d])))
        else:                       # level など
            ch.append(len(dimdict[d]))
    return tuple(ch)

# ───────── 2. blosc-LZ4 が使えるか確認 ─────────
def detect_blosc():
    fn = '__tmp__.nc'
    style = None
    try:
        ds = Dataset(fn, 'w', format='NETCDF4')
        ds.createDimension('x', 1)
        ds.createVariable('v', 'i2', ('x',),
                          compression='blosc',
                          compressor='lz4',
                          complevel=1)
        style = 'new'              # netcdf-c 4.9 以降の指定方法
    except TypeError:
        # 旧: compression='blosc_lz4'
        try:
            ds.createVariable('v', 'i2', ('x',),
                              compression='blosc_lz4',
                              complevel=1)
            style = 'old'
        except Exception:
            style = None
    finally:
        try: ds.close()
        except Exception: pass
        if os.path.exists(fn): os.remove(fn)
    return style

BLOSC_STYLE = detect_blosc()
if BLOSC_STYLE is None:
    raise RuntimeError('blosc-lz4 フィルタが利用できません')

# ───────── 3. 変数作成ヘルパ ─────────
def create_var(ds, name, dtype, dims, ch, fill_value):
    kw = dict(chunksizes=ch,
              complevel=3,
              shuffle=True,
              blosc_shuffle=1,
              fill_value=fill_value)
    if BLOSC_STYLE == 'new':
        kw.update(dict(compression='blosc', compressor='lz4'))
    else:
        kw.update(dict(compression='blosc_lz4'))
    return ds.createVariable(name, dtype, dims, **kw)

# ───────── 4. main ─────────
def main():
    src_dir = './128_128/nc_gsm7'
    dst_dir = './128_128/nc_gsm9'
    os.makedirs(dst_dir, exist_ok=True)

    for src in sorted(glob(os.path.join(src_dir, '*.nc'))):
        fn  = os.path.basename(src)
        dst = os.path.join(dst_dir, fn)

        if os.path.exists(dst):
            print('skip', fn)
            continue

        t0 = time.time()
        print('convert →', fn)

        with Dataset(src, 'r') as s, Dataset(dst, 'w', format='NETCDF4') as d:
            # --- 次元 ---
            for n, dim in s.dimensions.items():
                d.createDimension(n, len(dim) if not dim.isunlimited() else None)

            # --- global 属性 ---
            for a in s.ncattrs():
                if a != 'history':
                    d.setncattr(a, s.getncattr(a))
            hist_prev = s.getncattr('history') if 'history' in s.ncattrs() else ''
            d.setncattr('history',
                        f'{hist_prev} / {datetime.utcnow():%Y-%m-%dT%H:%MZ} blosc-lz4')

            # --- 座標変数（次元と同名の変数） ---
            for n, v in s.variables.items():
                if n in s.dimensions:
                    cv = create_var(d, n, v.datatype, v.dimensions,
                                    chunksizes(v, s.dimensions), fill_value=None)
                    cv[:] = v[:]
                    for a in v.ncattrs():
                        if a != '_FillValue':
                            cv.setncattr(a, v.getncattr(a))

            # --- データ変数 ---
            for n, v in s.variables.items():
                if n in s.dimensions:
                    continue
                fill_v = v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None
                ch     = chunksizes(v, s.dimensions)

                dv = create_var(d, n, v.datatype, v.dimensions, ch, fill_value=fill_v)
                dv[:] = v[:]

                for a in v.ncattrs():
                    if a != '_FillValue':
                        dv.setncattr(a, v.getncattr(a))

        print(f'   done  {time.time()-t0:.1f}s  '
              f'{os.path.getsize(dst)/1024/1024:.1f} MB')

if __name__ == '__main__':
    main()