export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="relative border-t border-white/5 py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-mute">
        <div className="flex items-center gap-2.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.svg" alt="" className="w-6 h-6" />
          <span className="text-white font-medium">Video Analysis Pro</span>
        </div>
        <p>
          © {year} 听风公司 (Tingfeng) · GPL v3 开源 ·{" "}
          <a
            href="https://github.com/lza6/Video-Analysis-Pro-python"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors underline-offset-4 hover:underline"
          >
            GitHub
          </a>
        </p>
      </div>
    </footer>
  );
}
