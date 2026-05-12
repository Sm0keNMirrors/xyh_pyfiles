import inspect
import importlib.util
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, render_template_string

def tool(name=None, description=None, param_md=None):
    """
    函数注册装饰器。

    用法：

    @tool(description="两个数相加")
    def add(a: float, b: float):
        return a + b
    """

    def decorator(fn):
        fn_name = name or fn.__name__

        FUNCTIONS[fn_name] = {
            "fn": fn,
            "name": fn_name,
            "description": description or inspect.getdoc(fn) or "",
            "source": fn.__module__,
            "param_md": param_md or "",
        }

        return fn

    return decorator




app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = BASE_DIR / "functions"
OUTPUT_DIR = BASE_DIR / "outputs"

PLUGIN_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FUNCTIONS = {}



# =========================
# 注册外部import函数
# =========================

from dataPost_processing.CMAQ_CombineWithISAM import CMAQ_CombineWithISAM
CMAQ_CombineWithISAM_PARAM_MD = """
### 参数说明
沿时间步 (TSTEP) 合并每日 CMAQ/CCTM NetCDF 文件。
```
参数
----------
start_date
    模拟起始日期，例如 "2024-11-30"。
CCTM_dir
    包含每日 CCTM/CMAQ nc 文件的文件夹。
Combine_file_outdir
    输出 nc 文件的路径。
GRIDDECfile_dir
    GRIDDESC 文件路径。
GRIDNAME
    GRIDDESC 文件中定义的网格名称。
CMAQISAM_version
    "v53" 或 "v54"。仅用于控制 PM25 组分计算和诊断因子 (diagnostic-factor) 逻辑。
combine_file_type
    需要合并的 CMAQ 输出类型，例如 "ACONC", "CONC", "SAACONC", "APMDIAG",
    "AELMO", "WETDEP1", "DRYDEP", "PA1", "IRR1", "AERODIAM", "AEROVIS"。
combine_vars
    需要合并的变量。示例：
    - ["O3", "NO2"] 用于普通变量。
    - ["PM25"] 用于从 PM 组分计算 PM25。
    - ["PM25", "O3"] 用于计算 PM25 并读取/合并 O3。
combine_isam_tags
    仅对 ISAM 输出文件设为 True。当为 True 时，请求的变量将通过基础物种进行匹配，
    并合并所有发现的标签，例如 O3_TAG1, O3_TAG2。当为 False 时，
    变量按精确名称读取。
pm25_components
    自定义 PM25 组分列表。如果为 None，则使用 CMAQISAM_version 对应的内置列表。
output_format
    PseudoNetCDF save() 函数使用的 NetCDF 输出格式。
dtype
    读取数据类型。对于大气科学数据，np.float32 比 np.float16 更安全。
ignore_missing_vars
    如果为 True，尽可能跳过缺失变量；否则报错。
combined_tagged_subs
    combine_vars 的向后兼容别名。
if_combinePA
    向后兼容标志。如果为 True，除非同时设置了 if_combine_special，
    否则 combine_file_type 将被强制设为 "PA1"。
if_combine_special
    向后兼容标志，例如 [True, "WETDEP1"]。
cores
    保留用于兼容性，在此稳定版本中未使用。

备注
-----
- PM25 是唯一会触发组分计算的变量。
- 其他变量直接按精确名称读取，或根据 combine_isam_tags 的设置进行 ISAM 标签搜索。
- 除非 combine_isam_tags=True，否则绝不会执行 ISAM 标签查找。
"""
tool(
    name="CMAQ_CombineWithISAM",
    description="运行 CMAQ 与 ISAM 数据合并处理",
    param_md=CMAQ_CombineWithISAM_PARAM_MD,
)(CMAQ_CombineWithISAM)


from dataPost_processing.Model_var_combine_by_time import Model_var_combine_by_time
Model_var_combine_by_time_PARAM_MD = """
将多个 WRF / CMAQ / NetCDF 文件中的指定变量按照时间维拼接。

    时间维不需要手动指定，默认使用目标变量的第一个维度。

    常见数据结构：
    CMAQ:
        var(TSTEP, LAY, ROW, COL)

    WRF:
        var(Time, bottom_top, south_north, west_east)
        var(Time, south_north, west_east)

    新增功能：
    1. verbose=True 时打印处理过程信息
    2. show_progress=True 时用 tqdm 显示文件处理进度
    3. gridcro2d_path 不为 None 时，从 CMAQ GRIDCRO2D 文件中读取 LAT / LON 加入输出结果

    Parameters
    ----------
    input_dir : str | Path
        输入文件目录。

    file_pattern : str
        文件匹配模式，例如：
        "ACONC_*.nc"
        "CCTM_ACONC_*.nc"
        "wrfout_d01_*"

    var_name : str
        目标变量名，例如：
        "O3", "PM25_TOT", "T2", "U", "V"

    output_path : str | Path | None
        输出 NetCDF 文件路径。
        如果为 None，则只返回 xr.Dataset，不写文件。

    recursive : bool
        是否递归搜索子目录。

    sort_regex : str | None
        从文件名中提取时间信息的正则表达式。
        例如 r"(\\d{8})"。

    sort_datetime_format : str | None
        文件名时间格式。
        例如 "%Y%m%d"。

    engine : str
        xarray 读取 NetCDF 的后端。
        常用 "netcdf4"。

    decode_times : bool
        是否让 xarray 自动解码时间。
        WRF / CMAQ 通常建议 False。

    mask_and_scale : bool
        是否应用 scale_factor / add_offset / missing_value。

    chunks : dict | None
        dask 分块参数。
        大文件可设置，例如 {"TSTEP": 24} 或 {"Time": 24}。

    time_source : str
        时间坐标来源：
        - "auto"  : 自动识别
        - "wrf"   : 使用 WRF Times
        - "cmaq"  : 使用 CMAQ TFLAG
        - "coord" : 使用已有坐标
        - "none"  : 使用连续整数时间索引

    output_var_name : str | None
        输出变量名。
        如果为 None，则沿用 var_name。

    zlib : bool
        输出 NetCDF 是否压缩。

    complevel : int
        压缩等级，1 到 9。

    gridcro2d_path : str | Path | None
        CMAQ GRIDCRO2D 文件路径。
        如果提供，则从该文件读取 LAT / LON 并加入最终输出。

    grid_lat_name : str
        GRIDCRO2D 中纬度变量名，默认 "LAT"。

    grid_lon_name : str
        GRIDCRO2D 中经度变量名，默认 "LON"。

    output_lat_name : str
        输出文件中纬度坐标名称，默认 "LAT"。

    output_lon_name : str
        输出文件中经度坐标名称，默认 "LON"。

    verbose : bool
        是否打印处理信息。

    show_progress : bool
        是否显示 tqdm 进度条。

    Returns
    -------
    xr.Dataset
        拼接后的单变量 Dataset。
"""
tool(
    name="Model_var_combine_by_time",
    description="按照时间步combine WRF CMAQ的输出变量",
    param_md=Model_var_combine_by_time_PARAM_MD,
)(Model_var_combine_by_time)




# =========================
# 内置函数示例
# =========================

@tool(description="两个数字相加")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="重复文本")
def repeat_text(text: str, times: int = 3) -> str:
    return text * times


@tool(description="格式化 JSON 字符串")
def pretty_json(json_text: str) -> dict:
    return json.loads(json_text)


@tool(description="列出本地目录文件")
def list_dir(path: str = ".") -> list:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    return [item.name for item in p.iterdir()]


# =========================
# 插件加载
# =========================

def load_plugins():
    """
    自动加载 functions/ 目录下的 .py 文件。

    插件文件示例：

    from webui_runner import tool

    @tool(description="打招呼")
    def hello(name: str):
        return f"Hello, {name}"
    """

    for py_file in PLUGIN_DIR.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"functions.{py_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)

            # 允许插件直接使用 tool，而不一定 import
            module.tool = tool

            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 自动注册插件里的公开函数
            for attr_name, obj in vars(module).items():
                if attr_name.startswith("_"):
                    continue
                if not inspect.isfunction(obj):
                    continue
                if obj.__module__ != module.__name__:
                    continue

                registered = any(v["fn"] is obj for v in FUNCTIONS.values())
                if not registered:
                    full_name = f"{py_file.stem}.{attr_name}"
                    FUNCTIONS[full_name] = {
                        "fn": obj,
                        "name": full_name,
                        "description": inspect.getdoc(obj) or "",
                        "source": module_name,
                    }

        except Exception:
            print(f"[Plugin load failed] {py_file}")
            traceback.print_exc()


# =========================
# 参数解析
# =========================

def annotation_to_type(annotation):
    if annotation is inspect._empty:
        return "str"

    origin = getattr(annotation, "__origin__", None)

    if annotation in [str, int, float, bool]:
        return annotation.__name__

    if annotation in [list, dict, tuple, set] or origin in [list, dict, tuple, set]:
        return "json"

    return "json"

def json_safe_default(value):
    if value is inspect._empty:
        return None

    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, type):
            return value.__name__

        if isinstance(value, Path):
            return str(value)

        return repr(value)

def function_schema(name, meta):
    fn = meta["fn"]
    sig = inspect.signature(fn)

    params = []

    for param_name, param in sig.parameters.items():
        if param.kind in [
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ]:
            continue

        has_default = param.default is not inspect._empty
        default = None if not has_default else json_safe_default(param.default)

        param_type = annotation_to_type(param.annotation)

        params.append({
            "name": param_name,
            "type": param_type,
            "required": not has_default,
            "default": default,
        })

    return {
        "name": name,
        "description": meta["description"],
        "source": meta["source"],
        "param_md": meta.get("param_md", ""),
        "params": params,
    }



def convert_value(raw, annotation, default):
    if raw in [None, ""]:
        if default is not inspect._empty:
            return default
        return raw

    if annotation is inspect._empty or annotation is str:
        return raw

    if annotation is int:
        return int(raw)

    if annotation is float:
        return float(raw)

    if annotation is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ["1", "true", "yes", "y", "on"]

    origin = getattr(annotation, "__origin__", None)

    if annotation in [list, dict, tuple, set] or origin in [list, dict, tuple, set]:
        value = json.loads(raw)
        if annotation is tuple:
            return tuple(value)
        if annotation is set:
            return set(value)
        return value

    # 其他复杂类型默认尝试 JSON
    try:
        return json.loads(raw)
    except Exception:
        return raw


def safe_json(value):
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, type):
            return value.__name__

        if isinstance(value, Path):
            return str(value)

        return repr(value)



# =========================
# API
# =========================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/functions")
def api_functions():
    schemas = [
        function_schema(name, meta)
        for name, meta in sorted(FUNCTIONS.items())
    ]
    return jsonify(schemas)


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True)

    name = payload.get("name")
    raw_params = payload.get("params", {})

    if name not in FUNCTIONS:
        return jsonify({
            "ok": False,
            "error": f"Function not found: {name}",
        }), 404

    meta = FUNCTIONS[name]
    fn = meta["fn"]
    sig = inspect.signature(fn)

    kwargs = {}

    try:
        for param_name, param in sig.parameters.items():
            if param.kind in [
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ]:
                continue

            raw = raw_params.get(param_name)
            kwargs[param_name] = convert_value(
                raw=raw,
                annotation=param.annotation,
                default=param.default,
            )

        result = fn(**kwargs)

        record = {
            "ok": True,
            "function": name,
            "params": kwargs,
            "result": safe_json(result),
            "result_type": type(result).__name__,
            "time": datetime.now().isoformat(timespec="seconds"),
        }

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace('.', '_')}_{uuid4().hex[:8]}.json"
        output_path = OUTPUT_DIR / filename

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

        record["output_file"] = str(output_path)

        return jsonify(record)

    except Exception as e:
        return jsonify({
            "ok": False,
            "function": name,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


# =========================
# WebUI 页面
# =========================

HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>Python Function WebUI Runner</title>
    <style>
        body {
            font-family: Arial, "Microsoft YaHei", sans-serif;
            margin: 0;
            background: #f6f7f9;
            color: #222;
        }

        .container {
            max-width: 980px;
            margin: 32px auto;
            padding: 24px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }

        h1 {
            margin-top: 0;
        }

        label {
            display: block;
            margin-top: 16px;
            font-weight: bold;
        }

        select,
        input,
        textarea {
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 14px;
        }

        textarea {
            min-height: 90px;
            font-family: Consolas, monospace;
        }

        button {
            margin-top: 20px;
            padding: 12px 18px;
            border: none;
            border-radius: 8px;
            background: #111;
            color: white;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            background: #333;
        }

        .desc {
            margin-top: 12px;
            padding: 12px;
            background: #f0f2f5;
            border-radius: 8px;
            white-space: pre-wrap;
        }

        .param-md-box {
            margin-top: 14px;
            padding: 14px 16px;
            background: #fffdf5;
            border: 1px solid #e6d8a8;
            border-radius: 10px;
            color: #333;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            overflow-x: auto;
        }

        .param-md-box-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #8a6500;
        }


        .param-card {
            margin-top: 14px;
            padding: 14px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: #fafafa;
        }

        .hint {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }

        pre {
            margin-top: 20px;
            padding: 16px;
            background: #111;
            color: #eee;
            border-radius: 10px;
            overflow: auto;
            min-height: 160px;
        }

        .error {
            color: #ffb4b4;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Python Function WebUI Runner</h1>

    <label>选择函数</label>
    <select id="functionSelect"></select>

    <div id="functionDesc" class="desc"></div>

    <div id="paramMdBox" class="param-md-box" style="display: none;"></div>

    <form id="paramForm"></form>

    <button onclick="runFunction()">运行函数</button>

    <pre id="output">等待运行...</pre>
</div>

<script>
let functions = [];

async function loadFunctions() {
    const res = await fetch("/api/functions");
    functions = await res.json();

    const select = document.getElementById("functionSelect");
    select.innerHTML = "";

    for (const fn of functions) {
        const option = document.createElement("option");
        option.value = fn.name;
        option.textContent = fn.name;
        select.appendChild(option);
    }

    select.addEventListener("change", renderParams);

    if (functions.length > 0) {
        renderParams();
    }
}

function currentFunction() {
    const name = document.getElementById("functionSelect").value;
    return functions.find(fn => fn.name === name);
}

function renderParams() {
    const fn = currentFunction();
    const desc = document.getElementById("functionDesc");
    const form = document.getElementById("paramForm");
    const paramMdBox = document.getElementById("paramMdBox");

    desc.textContent = `来源: ${fn.source}\n说明: ${fn.description || "无"}`;

    if (fn.param_md && fn.param_md.trim()) {
        paramMdBox.style.display = "block";
        paramMdBox.textContent = fn.param_md.trim();
    } else {
        paramMdBox.style.display = "none";
        paramMdBox.textContent = "";
    }

    form.innerHTML = "";

    for (const param of fn.params) {
        const card = document.createElement("div");
        card.className = "param-card";

        const label = document.createElement("label");
        label.textContent = `${param.name} (${param.type})${param.required ? " *" : ""}`;
        card.appendChild(label);

        let input;

        if (param.type === "json") {
            input = document.createElement("textarea");
            input.placeholder = "请输入 JSON，例如：[1, 2, 3] 或 {\\\"a\\\": 1}";
            if (param.default !== null && param.default !== undefined) {
                input.value = JSON.stringify(param.default, null, 2);
            }
        } else if (param.type === "bool") {
            input = document.createElement("select");

            const trueOption = document.createElement("option");
            trueOption.value = "true";
            trueOption.textContent = "true";

            const falseOption = document.createElement("option");
            falseOption.value = "false";
            falseOption.textContent = "false";

            input.appendChild(trueOption);
            input.appendChild(falseOption);

            if (param.default === false) {
                input.value = "false";
            }
        } else {
            input = document.createElement("input");
            input.type = "text";

            if (param.default !== null && param.default !== undefined) {
                input.value = param.default;
            }
        }

        input.name = param.name;
        input.dataset.type = param.type;

        card.appendChild(input);

        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = param.required ? "必填参数" : "可选参数，留空时使用默认值";
        card.appendChild(hint);

        form.appendChild(card);
    }
}

async function runFunction() {
    const fn = currentFunction();
    const form = document.getElementById("paramForm");
    const output = document.getElementById("output");

    const params = {};

    for (const input of form.querySelectorAll("input, textarea, select")) {
        params[input.name] = input.value;
    }

    output.textContent = "运行中...";

    try {
        const res = await fetch("/api/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: fn.name,
                params,
            }),
        });

        const data = await res.json();

        if (!data.ok) {
            output.innerHTML = `<span class="error">${escapeHtml(JSON.stringify(data, null, 2))}</span>`;
            return;
        }

        output.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
        output.innerHTML = `<span class="error">${escapeHtml(String(err))}</span>`;
    }
}

function escapeHtml(text) {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

loadFunctions();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    load_plugins()
    print("WebUI running at: http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
