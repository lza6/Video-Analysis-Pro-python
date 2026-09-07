# VAP Desktop — Electron 桌面壳

TingFeng Hermes 的 Electron 桌面壳。**不重写 UI**,复用现有 `webapp/`(Next.js)+ `src/web/serve.py`(FastAPI)。启动 Electron 主进程 → spawn `python src/web/serve.py` 子进程 → 探测就绪 → BrowserWindow 加载该端口。

## 目录

```
desktop/
├── package.json            # electron + electron-builder 依赖 + scripts
├── main.js                 # 主进程:单实例锁+托盘+窗口+子进程生命周期
├── preload.js               # contextBridge 暴露最小 API(getAppVersion/platform)
├── runtime-controller.js    # spawn python + stdout 探测端口 + health 兜底 + taskkill 清理
├── electron-builder.yml    # electron-builder 配置(nsis win target,venv 打包)
├── assets/
│   └── icon.png            # 占位图标(1x1 透明 PNG,发布前替换为真图标 256x256)
└── README.md
```

## 开发启动

前置:项目根 `venv/` 已建好依赖,`webapp/out/` 已构建(`cd webapp && npm run build`)。

```bash
cd desktop
npm install
npm start            # = electron .
```

启动后:
1. 主进程 spawn `venv/Scripts/python.exe src/web/serve.py --no-browser`
2. 读 stdout,正则 `http://127.0.0.1:(\d+)` 提取实际端口(serve.py 端口被占会自动跳)
3. 5s 内 stdout 没给端口 → 兜底轮询 8000-8019 的 `/api/health`
4. BrowserWindow `loadURL("http://127.0.0.1:<port>/")`
5. 托盘菜单(显示主窗口 / 后端端口 / 退出)
6. 关窗最小化到托盘,不退出;点退出才 kill 子进程

## 语法检查(不启动 Electron)

```bash
cd desktop
npm run check       # node --check 三个 .js
```

## 打包

```bash
cd desktop
npm run build:win   # electron-builder → dist/*.exe(nsis)
```

打包假设(供主控收口注意):
- `extraResources` 把项目根 `venv/` 与 `src/`、`config/prompts` 打进安装包,运行时 `runtime-controller.js` 用 `path.resolve(__dirname, "..")` 定位项目根——**打包后路径会变**,主控需在收口阶段把"项目根"从 `..` 改为 `process.resourcesPath`(见下"收口提示")。
- `webapp/out/` 经 `extraFiles` 复制到 `resources/webapp/out`,serve.py 的 `FRONTEND_DIST` 需指向它。

## 收口提示(给主控)

1. **路径定位**:开发期 `PROJECT_ROOT = path.resolve(__dirname, "..")` 指向仓库根;打包后 `__dirname` 在 `resources/app`,需改为:
   ```js
   const PROJECT_ROOT = app.isPackaged
     ? path.join(process.resourcesPath)
     : path.resolve(__dirname, "..");
   ```
   并相应把 `venv` / `src` / `webapp/out` 的路径前缀改为 `process.resourcesPath` 下。当前实现按开发期写,主控收口时统一处理。
2. **图标**:占位 1x1 PNG,发布前替换为 256x256 真图标(`assets/icon.png`),否则 nsis 用默认图标。
3. **venv 体积**:把整个 venv 打进安装包会很大(~数 GB,含 torch)。主控可考虑:精简 venv(只保留运行依赖)/ 用 PyInstaller 先把 `serve.py` 打成单 exe,再 `extraResources` 那个 exe。当前实现保留原样,主控决策。
4. **子进程端口探测**:依赖 serve.py stdout 打 `TingFeng Hermes Web UI: http://127.0.0.1:<port>`(已验证在 `src/web/serve.py:171`)。若主控后续改 serve.py 输出格式,需同步更新 `runtime-controller.js` 的 `READY_RE`。
5. **Windows GBK**:spawn 时设 `PYTHONIOENCODING=utf-8` + `PYTHONUNBUFFERED=1`,stdout 用 `chunk.toString("utf-8")`。已处理。

## 验证记录

- `node --check main.js` / `preload.js` / `runtime-controller.js`:零错(见 `npm run check`)
- `electron-builder.yml`:YAML 合法
