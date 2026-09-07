import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // react-hooks/set-state-in-effect 默认 error, 但本项目多处页面用
  // useEffect 挂载拉数据 + 定时轮询(refresh 是 useCallback, 内部用
  // Promise.allSettled 异步批量 setState, 非同步 effect body 连环 setState)。
  // 该规则对这些正当的 data-fetching effect 误报, 降为 warn 保留信号不阻断 CI。
  {
    rules: {
      "react-hooks/set-state-in-effect": "warn",
    },
  },
  globalIgnores([".next/**", "out/**", "build/**", "next-env.d.ts"]),
]);

export default eslintConfig;
