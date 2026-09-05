"use client";

import { useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

interface VideoPickerProps {
  file: File | null;
  localPath: string;
  onFile: (f: File | null) => void;
  onLocalPath: (p: string) => void;
  disabled?: boolean;
}

/**
 * 视频来源选择:拖拽/点击上传,或填本机路径(本地模式,隐私不出本机)。
 * 取代 PyQt6 的拖拽 drop frame。
 */
export function VideoPicker({
  file,
  localPath,
  onFile,
  onLocalPath,
  disabled,
}: VideoPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [mode, setMode] = useState<"upload" | "local">("upload");

  const handleFiles = (files: FileList | null) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <Card className="p-5">
      <div className="flex items-center gap-2 mb-4">
        <Button
          size="sm"
          variant={mode === "upload" ? "primary" : "chip"}
          onClick={() => setMode("upload")}
        >
          上传视频
        </Button>
        <Button
          size="sm"
          variant={mode === "local" ? "primary" : "chip"}
          onClick={() => setMode("local")}
        >
          本机路径
        </Button>
      </div>

      {mode === "upload" ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "rounded-card-sm border border-dashed px-6 py-10 text-center cursor-pointer transition-colors",
            dragOver
              ? "border-accent bg-accent/5"
              : "border-white/15 hover:border-white/30",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          {file ? (
            <div className="text-mist">
              <p className="font-medium text-white">{file.name}</p>
              <p className="text-xs text-mute mt-1">
                {(file.size / 1024 / 1024).toFixed(1)} MB · 点击或拖入可更换
              </p>
            </div>
          ) : (
            <div className="text-mute">
              <p className="text-white font-medium">拖入视频文件,或点击选择</p>
              <p className="text-xs mt-1">支持 MP4 / MKV / MOV / AVI / WebM 等</p>
            </div>
          )}
        </div>
      ) : (
        <div>
          <label className="block text-xs text-mute mb-1.5">
            本机视频绝对路径(不上传,直接读取)
          </label>
          <input
            type="text"
            value={localPath}
            disabled={disabled}
            onChange={(e) => onLocalPath(e.target.value)}
            placeholder="C:\videos\demo.mp4 或 /home/user/demo.mp4"
            className="w-full rounded-card-sm glass-chip px-4 py-3 text-sm text-white placeholder:text-mute/50 focus:outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
      )}
    </Card>
  );
}
