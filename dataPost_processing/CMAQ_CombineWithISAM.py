"""
Author: Yaohan Xian
GitHub: https://github.com/Sm0keNMirrors
Last update: 2026-04-28

Upgraded by ChatGPT:
- Support selectable CMAQ/CCTM output file types, including common CMAQ v5.3 files.
- Support user-provided combine variables.
- PM25 is calculated from PM2.5 components; other variables are read directly.
- ISAM tag search is only enabled when combine_isam_tags=True.
"""

import os
import re
import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import netCDF4 as nc
import PseudoNetCDF as pnc
from tqdm import tqdm


# =============================================================================
# CMAQ / ISAM PM2.5 settings
# =============================================================================

PM25_COMPONENTS_CMAQ_V53 = [
    "ASO4I", "ANO3I", "ANH4I", "ANAI", "ACLI", "AECI", "AOTHRI",
    "APOCI", "APNCOMI",

    "ASO4J", "ANO3J", "ANH4J", "ANAJ", "ACLJ", "AECJ",
    "AOTHRJ", "AFEJ", "ASIJ", "ATIJ", "ACAJ",
    "AMGJ", "AMNJ", "AALJ", "AKJ", "APOCJ", "APNCOMJ",
]

PM25_COMPONENTS_CMAQ_V54 = [
    "ASO4I", "ANO3I", "ANH4I", "ANAI", "ACLI", "AECI", "AOTHRI",
    "APOCI", "APNCOMI", "ALVOO1I", "ALVOO2I", "ASVOO1I", "ASVOO2I",
    "ALVPO1I", "ASVPO1I", "ASVPO2I",

    "ASO4J", "ANO3J", "ANH4J", "ANAJ", "ACLJ", "AXYL1J", "AXYL2J",
    "AXYL3J", "ATOL1J", "ATOL2J", "ATOL3J", "ABNZ1J", "ABNZ2J",
    "ABNZ3J", "AISO1J", "AISO2J", "AISO3J", "ATRP1J", "ATRP2J",
    "ASQTJ", "AALK1J", "AALK2J", "APAH1J", "APAH2J", "APAH3J",
    "AORGCJ", "AOLGBJ", "AOLGAJ", "ALVOO1J", "ALVOO2J", "ASVOO1J",
    "ASVOO2J", "ASVOO3J", "APCSOJ", "ALVPO1J", "ASVPO1J", "ASVPO2J",
    "ASVPO3J", "AIVPO1J", "AOTHRJ", "AFEJ", "ASIJ", "ATIJ", "ACAJ",
    "AMGJ", "AMNJ", "AALJ", "AKJ",

    "ASOIL", "ACORS", "ASEACAT", "ACLK", "ASO4K", "ANO3K", "ANH4K",
]

PM25_SETTINGS = {
    "v53": {
        "components": PM25_COMPONENTS_CMAQ_V53,
        "diag_file_type": "APMDIAG",
        "diag_vars": ["PM25AT", "PM25AC", "PM25CO"],
        "ait_factor": "PM25AT",
        "acc_factor": "PM25AC",
        "cor_factor": "PM25CO",
    },
    "v54": {
        "components": PM25_COMPONENTS_CMAQ_V54,
        "diag_file_type": "AELMO",
        "diag_vars": ["FPM25AIT", "FPM25ACC", "FPM25COR"],
        "ait_factor": "FPM25AIT",
        "acc_factor": "FPM25ACC",
        "cor_factor": "FPM25COR",
    },
}

# CMAQ v5.3 常见 CCTM 文件类型别名。匹配时既支持 CCTM_ACONC_*，
# 也支持 CCTM_SA_ACONC_* 这类 split 后需要拼接的名字。
CMAQ53_COMMON_FILE_TYPES = {
    "ACONC": ["ACONC"],
    "CONC": ["CONC"],
    "SAACONC": ["SAACONC", "SA_ACONC"],
    "APMDIAG": ["APMDIAG"],
    "AELMO": ["AELMO"],
    "AERODIAM": ["AERODIAM"],
    "AEROVIS": ["AEROVIS"],
    "SSEMIS": ["SSEMIS"],
    "DRYDEP": ["DRYDEP"],
    "WETDEP": ["WETDEP", "WETDEP1"],
    "WETDEP1": ["WETDEP1", "WETDEP"],
    "PA": ["PA", "PA1", "PA2", "PA3"],
    "PA1": ["PA1", "PA_1"],
    "PA2": ["PA2", "PA_2"],
    "IRR": ["IRR", "IRR1", "IRR2", "IRR3"],
    "IRR1": ["IRR1", "IRR_1"],
    "IRR2": ["IRR2", "IRR_2"],
    "DEPV": ["DEPV"],
    "DDEP": ["DDEP"],
    "WDEP": ["WDEP", "WETDEP", "WETDEP1"],
    "CGRID": ["CGRID"],
    "MEDIA_CONC": ["MEDIA_CONC", "MEDIACONC"],
}


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Convert None / str / sequence to a plain list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _norm_file_type(file_type: str) -> str:
    """Normalize a CMAQ file type token for matching."""
    return file_type.upper().replace("-", "_")


def _candidate_file_type_tokens(file_type: str) -> List[str]:
    """Return possible file type tokens for a requested CMAQ output type."""
    ft = _norm_file_type(file_type)
    tokens = CMAQ53_COMMON_FILE_TYPES.get(ft, [ft])
    out = []
    for t in tokens:
        t = _norm_file_type(t)
        out.extend([t, t.replace("_", "")])
    return sorted(set(out), key=len, reverse=True)


def _filename_tokens(filename: str) -> List[str]:
    """
    Extract useful matching tokens from a CCTM filename.

    Examples:
    CCTM_ACONC_v53_2024001.nc     -> ACONC
    CCTM_SA_ACONC_2024001.nc      -> SA, ACONC, SAACONC, SA_ACONC
    CCTM_WETDEP1_2024001.nc       -> WETDEP1
    """
    stem = Path(filename).name
    stem = stem[:-3] if stem.endswith(".nc") else stem
    parts = stem.split("_")
    tokens = []
    if len(parts) > 1:
        upper_parts = [_norm_file_type(p) for p in parts[1:]]
        tokens.extend(upper_parts)
        if len(upper_parts) >= 2:
            tokens.append(upper_parts[0] + upper_parts[1])
            tokens.append(upper_parts[0] + "_" + upper_parts[1])
        if len(upper_parts) >= 3:
            tokens.append(upper_parts[0] + upper_parts[1] + upper_parts[2])
            tokens.append(upper_parts[0] + "_" + upper_parts[1] + "_" + upper_parts[2])
    return sorted(set(tokens), key=len, reverse=True)


def _match_cmaq_file_type(filename: str, file_type: str) -> bool:
    """Whether filename looks like a requested CMAQ/CCTM output file type."""
    fn_tokens = set(_filename_tokens(filename))
    wanted = set(_candidate_file_type_tokens(file_type))
    return bool(fn_tokens & wanted)


def _extract_yyyymmdd_or_jdate(filename: str) -> Tuple[int, str]:
    """
    Return sortable date-like key from CMAQ filename.
    It supports YYYYMMDD and YYYYJJJ patterns. If no date exists, filename is used.
    """
    base = Path(filename).name
    m = re.search(r"(20\d{5}|19\d{5}|20\d{6}|19\d{6})", base)
    if not m:
        return (999999999, base)
    txt = m.group(1)
    return (int(txt), base)


def _find_cmaq_files(
    cctm_dir: Union[str, Path],
    file_type: str,
    required_vars: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Find and sort CCTM nc files for a given CMAQ output type.

    If required_vars is provided, files that do not contain any of these variables
    are still kept out. This is useful when APMDIAG/AELMO variants exist.
    """
    cctm_dir = Path(cctm_dir)
    files = [
        f.name for f in cctm_dir.iterdir()
        if f.is_file() and f.name.endswith(".nc") and _match_cmaq_file_type(f.name, file_type)
    ]
    files = sorted(files, key=_extract_yyyymmdd_or_jdate)

    if required_vars:
        required_vars = set(required_vars)
        filtered = []
        for f in files:
            try:
                with nc.Dataset(cctm_dir / f) as ds:
                    if required_vars & set(ds.variables.keys()):
                        filtered.append(f)
            except OSError:
                continue
        files = filtered

    return files


def _read_and_concat(
    cctm_dir: Union[str, Path],
    files: Sequence[str],
    varname: str,
    dtype=np.float32,
    ignore_missing: bool = False,
) -> np.ndarray:
    """
    Read varname from daily CMAQ files and concatenate along TSTEP.

    The old function concatenated all daily files directly; this keeps that behavior.
    """
    cctm_dir = Path(cctm_dir)
    arrays = []
    for fname in files:
        with nc.Dataset(cctm_dir / fname) as ds:
            if varname not in ds.variables:
                if ignore_missing:
                    continue
                raise KeyError(f"变量 {varname} 不存在于文件 {fname}")
            arrays.append(np.asarray(ds.variables[varname][:], dtype=dtype))

    if not arrays:
        raise KeyError(f"在指定文件中没有读取到变量 {varname}")

    return np.concatenate(arrays, axis=0)


def _get_base_and_tag(varname: str, is_pa_file: bool = False) -> Tuple[str, str]:
    """
    Return base species name and ISAM tag suffix/prefix.

    Normal ISAM SAACONC usually uses SPEC_TAG, e.g. O3_BCON.
    PA files in the original script appeared to use TAG_SPEC.
    """
    parts = varname.split("_")
    if len(parts) < 2:
        return varname, ""

    if is_pa_file:
        tag = parts[0]
        base = "_".join(parts[1:])
        return base, f"{tag}_"

    base = parts[0]
    tag = "_".join(parts[1:])
    return base, f"_{tag}"


def _select_variables_from_template(
    template_file: Union[str, Path],
    requested_vars: Sequence[str],
    combine_isam_tags: bool = False,
    pm25_components: Optional[Sequence[str]] = None,
    is_pa_file: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Select variables from the first file.

    - combine_isam_tags=False:
      variables are exact names, e.g. ["O3", "NO2"].
    - combine_isam_tags=True:
      variables are selected by base species, e.g. requested "O3" matches O3_TAG.
      If "PM25" is requested, PM2.5 components are matched by base species.
    """
    with nc.Dataset(template_file) as ds:
        all_vars = [v for v in ds.variables.keys() if v != "TFLAG"]

    requested_vars = list(requested_vars)
    requested_base = set(requested_vars)
    if "PM25" in requested_base and pm25_components:
        requested_base.remove("PM25")
        requested_base.update(pm25_components)

    selected = []
    tags = []

    if combine_isam_tags:
        for v in all_vars:
            base, tag = _get_base_and_tag(v, is_pa_file=is_pa_file)
            if base in requested_base:
                selected.append(v)
            if tag and tag not in tags:
                tags.append(tag)
    else:
        selected = [v for v in requested_base if v in all_vars]

    return selected, tags


def _copy_ncattrs_from_source(
    target_var,
    source_file: Union[str, Path],
    source_varname: str,
    fallback_units: str = "unknown",
):
    """Copy variable attributes from a source CMAQ file when possible."""
    try:
        with nc.Dataset(source_file) as ds:
            src = ds.variables[source_varname]
            attrs = {a: src.getncattr(a) for a in src.ncattrs()}
            if attrs:
                target_var.setncatts(attrs)
                return
    except Exception:
        pass

    target_var.setncatts({
        "units": fallback_units,
        "long_name": source_varname[:16].ljust(16),
        "var_desc": source_varname,
    })


def _sum_existing_vars(var_dict: Dict[str, np.ndarray], names: Sequence[str], shape) -> np.ndarray:
    """Sum variables that exist in var_dict; missing names are ignored."""
    out = np.zeros(shape, dtype=np.float32)
    for name in names:
        if name in var_dict:
            out += np.asarray(var_dict[name], dtype=np.float32)
    return out


def _component_groups_for_pm25(varnames: Iterable[str], tag: str = "") -> Tuple[List[str], List[str], List[str]]:
    """Split PM component variable names into Aitken, accumulation, coarse groups."""
    ai, aj, ak = [], [], []
    for v in varnames:
        base, var_tag = _get_base_and_tag(v, is_pa_file=False)
        if tag and var_tag != tag:
            continue
        if base.startswith("A") and base.endswith("I"):
            ai.append(v)
        elif base.startswith("A") and base.endswith("J"):
            aj.append(v)
        elif (base.startswith("A") and base.endswith("K")) or base in ["ASOIL", "ACORS", "ASEACAT"]:
            ak.append(v)
    return ai, aj, ak


def _calculate_pm25(
    all_vars: Dict[str, np.ndarray],
    settings: Dict[str, Union[str, List[str]]],
    tag: str = "",
) -> np.ndarray:
    """
    Calculate PM25 from PM components.

    For CMAQ/ISAM v5.3:
        PM25 = sum(I-mode) * PM25AT + sum(J-mode) * PM25AC

    For v5.4:
        PM25 = sum(I-mode) * FPM25AIT + sum(J-mode) * FPM25ACC + sum(K/coarse) * FPM25COR
    """
    data_shape = next(iter(all_vars.values())).shape
    ai_names, aj_names, ak_names = _component_groups_for_pm25(all_vars.keys(), tag=tag)

    ai = _sum_existing_vars(all_vars, ai_names, data_shape)
    aj = _sum_existing_vars(all_vars, aj_names, data_shape)
    ak = _sum_existing_vars(all_vars, ak_names, data_shape)

    ait_factor = np.asarray(all_vars[settings["ait_factor"]], dtype=np.float32)
    acc_factor = np.asarray(all_vars[settings["acc_factor"]], dtype=np.float32)

    pm25 = ai * ait_factor + aj * acc_factor

    cor_factor_name = settings.get("cor_factor")
    if cor_factor_name in all_vars:
        cor_factor = np.asarray(all_vars[cor_factor_name], dtype=np.float32)
        pm25 = pm25 + ak * cor_factor

    return pm25.astype(np.float32)


def CMAQ_CombineWithISAM(
    start_date: str = "YYYY-MM-DD",
    CCTM_dir: str = "",
    Combine_file_outdir: str = "",
    GRIDDECfile_dir: str = "",
    GRIDNAME: str = "",
    CMAQISAM_version: str = "v53",

    # New recommended arguments
    combine_file_type: str = "SAACONC",
    combine_vars: Optional[Sequence[str]] = None,
    combine_isam_tags: bool = False,
    pm25_components: Optional[Sequence[str]] = None,
    output_format: str = "NETCDF3_CLASSIC",
    dtype=np.float32,
    ignore_missing_vars: bool = False,

    # Backward-compatible arguments
    combined_tagged_subs: Optional[Sequence[str]] = None,
    if_combinePA: bool = False,
    if_combine_special: Sequence[Union[bool, str]] = (False, ""),

    # Kept for compatibility; this version reads sequentially for stability.
    cores: int = 1,
):
    """
    Combine daily CMAQ/CCTM NetCDF files along TSTEP.

    Parameters
    ----------
    start_date
        Simulation start date, e.g. "2024-11-30".
    CCTM_dir
        Folder containing daily CCTM/CMAQ nc files.
    Combine_file_outdir
        Output nc path.
    GRIDDECfile_dir
        GRIDDESC file path.
    GRIDNAME
        Grid name in GRIDDESC.
    CMAQISAM_version
        "v53" or "v54". It only controls PM25 component and diagnostic-factor logic.
    combine_file_type
        CMAQ output type to combine, e.g. "ACONC", "CONC", "SAACONC", "APMDIAG",
        "AELMO", "WETDEP1", "DRYDEP", "PA1", "IRR1", "AERODIAM", "AEROVIS".
    combine_vars
        Variables to combine. Example:
        - ["O3", "NO2"] for ordinary variables.
        - ["PM25"] to calculate PM25 from PM components.
        - ["PM25", "O3"] to calculate PM25 and read/combine O3.
    combine_isam_tags
        Only set True for ISAM output files. When True, requested variables are
        matched by base species and all discovered tags are combined, e.g.
        O3_TAG1, O3_TAG2. When False, variables are read by exact variable name.
    pm25_components
        Custom PM25 component list. If None, use the built-in CMAQISAM_version list.
    output_format
        NetCDF output format used by PseudoNetCDF save().
    dtype
        Read dtype. np.float32 is safer than np.float16 for atmospheric data.
    ignore_missing_vars
        If True, missing variables are skipped where possible; otherwise raise error.
    combined_tagged_subs
        Backward-compatible alias for combine_vars.
    if_combinePA
        Backward-compatible flag. If True, combine_file_type is forced to "PA1"
        unless if_combine_special is also set.
    if_combine_special
        Backward-compatible flag like [True, "WETDEP1"].
    cores
        Kept for compatibility. Not used in this stable version.

    Notes
    -----
    - PM25 is the only variable that triggers component calculation.
    - Other variables are directly read by exact name or by ISAM tag search,
      depending on combine_isam_tags.
    - ISAM tag lookup is never performed unless combine_isam_tags=True.
    """
    if not CCTM_dir:
        raise ValueError("CCTM_dir 不能为空")
    if not Combine_file_outdir:
        raise ValueError("Combine_file_outdir 不能为空")
    if not GRIDDECfile_dir:
        raise ValueError("GRIDDECfile_dir 不能为空")
    if not GRIDNAME:
        raise ValueError("GRIDNAME 不能为空")

    # Backward compatibility:
    # old calls used combined_tagged_subs=["O3", "PM25"].
    if combine_vars is None:
        combine_vars = list(combined_tagged_subs or [])
    else:
        combine_vars = list(combine_vars)

    if not combine_vars:
        raise ValueError("请通过 combine_vars 或 combined_tagged_subs 指定需要 combine 的变量")

    if if_combine_special and bool(if_combine_special[0]):
        combine_file_type = str(if_combine_special[1])
    elif if_combinePA:
        combine_file_type = "PA1"

    version = CMAQISAM_version.lower()
    if version not in PM25_SETTINGS and "PM25" in combine_vars:
        raise ValueError("计算 PM25 时 CMAQISAM_version 必须是 'v53' 或 'v54'")

    pm_settings = PM25_SETTINGS.get(version)
    if pm25_components is None and pm_settings:
        pm25_components = pm_settings["components"]

    cctm_dir = Path(CCTM_dir)

    # Main files
    to_combine_files = _find_cmaq_files(cctm_dir, combine_file_type)
    if not to_combine_files:
        raise FileNotFoundError(f"未找到类型为 {combine_file_type} 的 nc 文件：{CCTM_dir}")

    # Diagnostic files for PM25
    diag_files = []
    diag_vars = []
    if "PM25" in combine_vars:
        diag_file_type = pm_settings["diag_file_type"]
        diag_vars = list(pm_settings["diag_vars"])
        # If the chosen file type itself is the diagnostic file, reuse it.
        if _norm_file_type(combine_file_type) == _norm_file_type(diag_file_type):
            diag_files = to_combine_files
        else:
            diag_files = _find_cmaq_files(cctm_dir, diag_file_type, required_vars=diag_vars)

        if not diag_files:
            raise FileNotFoundError(
                f"需要计算 PM25，但未找到 PM 诊断文件 {diag_file_type}，"
                f"或文件内缺少诊断变量 {diag_vars}"
            )

    # Select variables
    template_file = cctm_dir / to_combine_files[0]
    selected_vars, isam_tags = _select_variables_from_template(
        template_file=template_file,
        requested_vars=combine_vars,
        combine_isam_tags=combine_isam_tags,
        pm25_components=pm25_components,
        is_pa_file=if_combinePA,
    )

    # Add diagnostic variables when PM25 is requested
    for v in diag_vars:
        if v not in selected_vars:
            selected_vars.append(v)

    if not selected_vars:
        raise ValueError(
            f"未在 {template_file.name} 中找到需要 combine 的变量。"
            f"combine_vars={combine_vars}, combine_isam_tags={combine_isam_tags}"
        )

    # Split main vars and diagnostic vars
    main_vars = [v for v in selected_vars if v not in diag_vars]
    all_vars_data: Dict[str, np.ndarray] = {}

    print(f"[INFO] combine_file_type = {combine_file_type}")
    print(f"[INFO] main files = {len(to_combine_files)}")
    print(f"[INFO] variables from main files = {main_vars}")
    if diag_vars:
        print(f"[INFO] PM diagnostic file type = {pm_settings['diag_file_type']}")
        print(f"[INFO] diagnostic files = {len(diag_files)}")
        print(f"[INFO] diagnostic variables = {diag_vars}")
    if combine_isam_tags:
        print(f"[INFO] ISAM tags = {isam_tags}")

    for v in tqdm(main_vars, desc=f"读取并合并 {combine_file_type} 变量"):
        try:
            all_vars_data[v] = _read_and_concat(
                cctm_dir, to_combine_files, v, dtype=dtype, ignore_missing=ignore_missing_vars
            )
        except KeyError:
            if ignore_missing_vars:
                print(f"[WARN] 跳过缺失变量：{v}")
                continue
            raise

    for v in tqdm(diag_vars, desc="读取并合并 PM25 诊断变量"):
        all_vars_data[v] = _read_and_concat(
            cctm_dir, diag_files, v, dtype=dtype, ignore_missing=False
        )

    if not all_vars_data:
        raise ValueError("没有成功读取任何变量，无法生成 combine 文件")

    # Calculate output TSTEP number from actual concatenated data, not file count.
    n_tstep = next(iter(all_vars_data.values())).shape[0]
    yyyyjjj = datetime.datetime.strftime(pd.to_datetime(str(start_date)), "%Y%j")

    gf = pnc.pncopen(
        GRIDDECfile_dir,
        GDNAM=GRIDNAME,
        format="griddesc",
        SDATE=int(yyyyjjj),
        TSTEP=10000,
        withcf=False,
    )
    gf.updatetflag(overwrite=True)
    combine_nc = gf.sliceDimensions(TSTEP=[0] * n_tstep)

    # Create variables copied from source
    for v in tqdm(all_vars_data, desc="写入 combine nc 变量"):
        if v in combine_nc.variables:
            del combine_nc.variables[v]
        var = combine_nc.createVariable(v, "f", ("TSTEP", "LAY", "ROW", "COL"))

        src_file = cctm_dir / (diag_files[0] if v in diag_vars else to_combine_files[0])
        _copy_ncattrs_from_source(var, src_file, v)
        var[:, :, :, :] = all_vars_data[v][:, :, :, :]

    # PM25 calculation
    if "PM25" in combine_vars:
        if combine_isam_tags:
            tags_to_calculate = isam_tags
        else:
            tags_to_calculate = [""]

        for tag in tqdm(tags_to_calculate, desc="计算 PM25"):
            pm25_data = _calculate_pm25(all_vars_data, pm_settings, tag=tag)

            pm25_name = f"PM25{tag}"
            if pm25_name in combine_nc.variables:
                del combine_nc.variables[pm25_name]
            pm25_var = combine_nc.createVariable(pm25_name, "f", ("TSTEP", "LAY", "ROW", "COL"))
            pm25_var.long_name = pm25_name[:16].ljust(16)
            pm25_var.units = "ug/m-3"
            pm25_var.var_desc = pm25_name
            pm25_var[:, :, :, :] = pm25_data

    # Final metadata
    if "DUMMY" in combine_nc.variables:
        del combine_nc.variables["DUMMY"]

    combine_nc.updatetflag(tstep=10000, overwrite=True)
    if hasattr(combine_nc, "VAR-LIST"):
        delattr(combine_nc, "VAR-LIST")
    combine_nc.updatemeta()

    output_name = str(Combine_file_outdir)
    combine_nc.save(output_name, format=output_format)
    combine_nc.close()

    print(f"[DONE] saved: {output_name}")
    return output_name


# =============================================================================
# Usage examples
# =============================================================================
if __name__ == "__main__":

    # Example 1: 普通 CMAQ ACONC，直接读取变量，不进行 ISAM tag 查找
    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\aconc\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\combine_aconc.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combine_file_type="ACONC",
    #     combine_vars=["O3", "NO2", "SO2"],
    #     combine_isam_tags=False,
    # )

    # Example 2: 普通 CMAQ PM25，从 PM 组分 + APMDIAG 诊断参数计算
    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\aconc\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\combine_pm25.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combine_file_type="ACONC",
    #     combine_vars=["PM25"],
    #     combine_isam_tags=False,
    # )

    # Example 3: ISAM SAACONC，查找并 combine 所有 O3 tag
    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\cctm_reo3\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\combine_isam_o3.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combine_file_type="SAACONC",
    #     combine_vars=["O3"],
    #     combine_isam_tags=True,
    # )

    # Example 4: ISAM SAACONC，按 tag 计算 PM25
    # CMAQ_CombineWithISAM(
    #     start_date="2024-11-30",
    #     CCTM_dir=r"E:\WCAS_serverfiles\cctm\cctm_repm25\\",
    #     Combine_file_outdir=r"E:\WCAS_serverfiles\cctm\combine_isam_pm25.nc",
    #     GRIDDECfile_dir=r"E:\WCAS_serverfiles\GRIDDESC",
    #     GRIDNAME="CDsvSA_d03",
    #     CMAQISAM_version="v53",
    #     combine_file_type="SAACONC",
    #     combine_vars=["PM25"],
    #     combine_isam_tags=True,
    # )

    pass
