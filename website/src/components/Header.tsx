"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";

const LINKS = [
  { href: "#features", label: "功能" },
  { href: "#how", label: "工作原理" },
  { href: "#download", label: "下载" },
  { href: "#faq", label: "常见问题" },
];

export default function Header() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const prefersReduced = useReducedMotion();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Esc 关闭移动菜单
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header
      className="fixed top-0 inset-x-0 z-40 flex justify-center px-4 pt-4"
      role="banner"
    >
      <motion.nav
        initial={prefersReduced ? false : { y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className={`glass glass-edge relative z-40 w-full max-w-6xl rounded-full px-5 sm:px-7 py-3 flex items-center justify-between transition-shadow duration-300 ${
          scrolled ? "shadow-[0_10px_40px_-15px_oklch(0_0_0/0.5)]" : ""
        }`}
        aria-label="主导航"
      >
        <a
          href="#top"
          className="flex items-center gap-2.5 text-white font-bold tracking-tight"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.svg" alt="" className="w-7 h-7" />
          <span className="text-[15px]">Video Analysis Pro</span>
        </a>

        <ul className="hidden md:flex items-center gap-7 text-sm">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a
                href={l.href}
                className="text-mute hover:text-white transition-colors"
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="hidden md:block">
          <a
            href="#download"
            className="glass-chip glass-edge rounded-full px-5 py-2 text-sm text-white hover:border-white/30 transition-colors"
          >
            立即下载
          </a>
        </div>

        <button
          onClick={() => setOpen((v) => !v)}
          className="md:hidden glass-chip rounded-full w-10 h-10 flex items-center justify-center text-white"
          aria-label={open ? "关闭菜单" : "打开菜单"}
          aria-expanded={open}
        >
          <div className="space-y-1.5">
            <span
              className={`block h-0.5 w-5 bg-white transition-transform ${
                open ? "translate-y-2 rotate-45" : ""
              }`}
            />
            <span
              className={`block h-0.5 w-5 bg-white transition-opacity ${
                open ? "opacity-0" : ""
              }`}
            />
            <span
              className={`block h-0.5 w-5 bg-white transition-transform ${
                open ? "-translate-y-2 -rotate-45" : ""
              }`}
            />
          </div>
        </button>
      </motion.nav>

      <AnimatePresence>
        {open && (
          <>
            {/* 点击外部关闭：全屏透明 overlay */}
            <motion.div
              key="overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="md:hidden fixed inset-0 z-30"
              onClick={() => setOpen(false)}
              aria-hidden="true"
            />
            <motion.div
              key="menu"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="md:hidden absolute top-20 inset-x-4 z-40 glass-strong glass-edge rounded-2xl p-4"
              onClick={(e) => e.stopPropagation()}
            >
              <ul className="flex flex-col gap-1">
                {LINKS.map((l) => (
                  <li key={l.href}>
                    <a
                      href={l.href}
                      onClick={() => setOpen(false)}
                      className="block px-4 py-3 text-white/90 hover:bg-white/5 rounded-lg transition-colors"
                    >
                      {l.label}
                    </a>
                  </li>
                ))}
                <li>
                  <a
                    href="#download"
                    onClick={() => setOpen(false)}
                    className="block px-4 py-3 mt-1 text-center rounded-lg bg-accent-2 text-white font-medium"
                  >
                    立即下载
                  </a>
                </li>
              </ul>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </header>
  );
}
