import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: true,
    port: 5173,
    fs: { allow: [".."] },
  },
  build: {
    target: "es2022",
  },
});
