#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gsm7 → gsm10 変換
  ・offset / scale / sqrt で正規化（float32 のまま）
  ・int16 量子化は行わない
  ・チャンク = (1,128,128)
  ・圧縮は blosc-LZ4 のみ
出力 : ./128_128/nc_gsm10
"""
import os, re, time
from glob import glob
from datetime import datetime
import numpy as np
from netCDF4 import Dataset

# ───────── 1. 正規化ルール ─────────
def get_norm_rule(name: str):
    if re.fullmatch(r'surface_prmsl|MSLP|pres|surface_sp', name):
        return (-100_000., 0.0001, False)
    if name in ('surface_u10', 'surface_v10'):
        return (0., 0.1, False)
    if re.search(r'(?:_T$|_t2m$|_T2m$)', name) and 'Td' not in name:
        return (-273.15, 0.02, False)
    if re.search(r'Td|_rh|_RH', name, re.IGNORECASE):
        return (0., 0.01, False)
    if re.fullmatch(r'level\d+_u', name) or re.fullmatch(r'level\d+_v', name):
        return (0., 0.05, False)
    gh = {'level925_gh': 0., 'level850_gh': -1000.,
          'level700_gh': -3000., 'level500_gh': -5000.,
          'level300_gh': -9000.}
    if name in gh:
        return (gh[name], 0.001, False)
    return None

# ───────── 2. 正規化のみ（量子化なし）─────────
def normalize_to_f32(arr, offset, scale, sqrt):          # ←★ 新関数
    a = arr.astype(np.float32)
    if sqrt:
        a = np.where(a < 0, 0, np.sqrt(a))
    return (a + offset) * scale                          # float32 のまま返す

# ───────── 3. チャンク ─────────
def chunksizes(var, dimdict):
    ch = []
    for d in var.dimensions:
        if d == 'time':
            ch.append(1)
        elif d in ('lat', 'lon'):
            ch.append(min(128, len(dimdict[d])))
        else:
            ch.append(len(dimdict[d]))
    return tuple(ch)

# ───────── 4. blosc-LZ4 利用可否 ─────────
def detect_blosc():
    fn = '__tmp__.nc'
    style = None
    try:
        ds = Dataset(fn, 'w', format='NETCDF4')
        ds.createDimension('x', 1)
        ds.createVariable('v', 'i2', ('x',),
                          compression='blosc', compressor='lz4', complevel=1)
        style = 'new'
    except TypeError:
        try:
            ds.createVariable('v', 'i2', ('x',),
                              compression='blosc_lz4', complevel=1)
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

def create_var(ds, name, dtype, dims, ch, fill_value):
    kw = dict(chunksizes=ch,
              complevel=3,
              shuffle=True,
              blosc_shuffle=1,
              fill_value=fill_value)
    if BLOSC_STYLE == 'new':
        kw.update(dict(compression='blosc',  compressor='lz4'))
    else:
        kw.update(dict(compression='blosc_lz4'))
    return ds.createVariable(name, dtype, dims, **kw)

# ───────── 5. main ─────────
def main():
    src_dir = './128_128/nc_gsm7'
    dst_dir = './128_128/nc_gsm10'          # ←★ 出力フォルダを変更
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

            # 次元
            for n, dim in s.dimensions.items():
                d.createDimension(n, len(dim) if not dim.isunlimited() else None)

            # global 属性
            for a in s.ncattrs():
                if a != 'history':
                    d.setncattr(a, s.getncattr(a))
            hist = s.getncattr('history') if 'history' in s.ncattrs() else ''
            d.setncattr('history',
                        f'{hist} / {datetime.utcnow():%Y-%m-%dT%H:%MZ} normalized-f32-lz4')

            # 座標変数
            for n, v in s.variables.items():
                if n in s.dimensions:
                    cv = create_var(d, n, v.datatype, v.dimensions,
                                    chunksizes(v, s.dimensions), fill_value=None)
                    cv[:] = v[:]
                    for a in v.ncattrs():
                        if a != '_FillValue':
                            cv.setncattr(a, v.getncattr(a))

            # データ変数
            for n, v in s.variables.items():
                if n in s.dimensions:
                    continue

                rule = get_norm_rule(n)
                ch   = chunksizes(v, s.dimensions)

                if rule is None:
                    # そのままコピー
                    fv = v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None
                    dv = create_var(d, n, v.datatype, v.dimensions, ch, fill_value=fv)
                    dv[:] = v[:]
                    for a in v.ncattrs():
                        if a != '_FillValue':
                            dv.setncattr(a, v.getncattr(a))

                else:
                    # 正規化のみ（float32）
                    off, sc, sq = rule
                    data32 = normalize_to_f32(v[:], off, sc, sq)   # ←★ 量子化なし
                    dv = create_var(d, n, 'f4', v.dimensions, ch,  # ←★ dtype は 'f4'
                                    fill_value=np.float32(np.nan))
                    dv[:] = data32.astype(np.float32)
                    for a in v.ncattrs():
                        if a != '_FillValue':
                            dv.setncattr(a, v.getncattr(a))
                    dv.setncattr('normalized',
                                 f'offset={off}, scale={sc}, sqrt={sq}, stored=float32')

        print(f'   done  {time.time()-t0:.1f}s  '
              f'{os.path.getsize(dst)/1024/1024:.1f} MB')

if __name__ == '__main__':
    main()