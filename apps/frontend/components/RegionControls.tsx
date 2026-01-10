"use client";

import { useState } from "react";
import type { RegionInfo } from "@/lib/api";

interface RegionState {
  visible: boolean;
  opacity: number;
}

interface RegionControlsProps {
  regions: RegionInfo[];
  regionStates: Record<string, RegionState>;
  onVisibilityChange: (regionName: string, visible: boolean) => void;
  onOpacityChange: (regionName: string, opacity: number) => void;
  onIsolate: (regionName: string | null) => void;
  isolatedRegion: string | null;
  hasTumor: boolean;
  atlasRegistered: boolean;
  inputType?: "segmentation" | "intensity";
}

export default function RegionControls({
  regions,
  regionStates,
  onVisibilityChange,
  onOpacityChange,
  onIsolate,
  isolatedRegion,
  hasTumor,
  atlasRegistered,
  inputType,
}: RegionControlsProps) {
  const [expanded, setExpanded] = useState(true);

  const handleToggleAll = (visible: boolean) => {
    regions.forEach((region) => {
      onVisibilityChange(region.name, visible);
    });
  };

  return (
    <div className="bg-black/80 backdrop-blur-sm rounded-lg overflow-hidden w-64">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-white/5"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-white font-medium text-sm">Brain Regions</span>
          {atlasRegistered && (
            <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
              Atlas
            </span>
          )}
          {inputType === "segmentation" && (
            <span className="text-xs bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded">
              Segmentation
            </span>
          )}
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {expanded && (
        <div className="border-t border-white/10">
          {/* Show/Hide All Buttons */}
          <div className="flex gap-2 px-4 py-2 border-b border-white/10">
            <button
              onClick={() => handleToggleAll(true)}
              className="flex-1 text-xs bg-white/10 hover:bg-white/20 text-white py-1 px-2 rounded transition-colors"
            >
              Show All
            </button>
            <button
              onClick={() => handleToggleAll(false)}
              className="flex-1 text-xs bg-white/10 hover:bg-white/20 text-white py-1 px-2 rounded transition-colors"
            >
              Hide All
            </button>
            {isolatedRegion && (
              <button
                onClick={() => onIsolate(null)}
                className="text-xs bg-blue-500/30 hover:bg-blue-500/50 text-blue-300 py-1 px-2 rounded transition-colors"
              >
                Reset
              </button>
            )}
          </div>

          {/* Region List */}
          <div className="max-h-80 overflow-y-auto">
            {regions.map((region) => {
              const state = regionStates[region.name] || { visible: true, opacity: region.opacity };
              const isIsolated = isolatedRegion === region.name;
              const colorStyle = `rgb(${region.color[0]}, ${region.color[1]}, ${region.color[2]})`;

              return (
                <div
                  key={region.name}
                  className={`px-4 py-2 border-b border-white/5 ${isIsolated ? "bg-blue-500/10" : ""}`}
                >
                  {/* Region Header */}
                  <div className="flex items-center gap-2 mb-1">
                    {/* Visibility Toggle */}
                    <button
                      onClick={() => onVisibilityChange(region.name, !state.visible)}
                      className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                        state.visible
                          ? "bg-white border-white"
                          : "bg-transparent border-gray-500"
                      }`}
                    >
                      {state.visible && (
                        <svg className="w-3 h-3 text-gray-900" fill="currentColor" viewBox="0 0 20 20">
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </button>

                    {/* Color Indicator */}
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: colorStyle }}
                    />

                    {/* Label */}
                    <span className="text-white text-sm flex-1">{region.label}</span>

                    {/* Tumor Badge */}
                    {region.name === "tumor" && hasTumor && (
                      <span className="text-xs bg-red-500/30 text-red-300 px-1.5 py-0.5 rounded">
                        Detected
                      </span>
                    )}

                    {/* Isolate Button */}
                    <button
                      onClick={() => onIsolate(isIsolated ? null : region.name)}
                      className={`text-xs px-1.5 py-0.5 rounded transition-colors ${
                        isIsolated
                          ? "bg-blue-500 text-white"
                          : "bg-white/10 text-gray-400 hover:bg-white/20 hover:text-white"
                      }`}
                      title="Isolate this region"
                    >
                      {isIsolated ? "Exit" : "Solo"}
                    </button>
                  </div>

                  {/* Opacity Slider */}
                  {state.visible && (
                    <div className="flex items-center gap-2 pl-6">
                      <span className="text-xs text-gray-500 w-12">Opacity</span>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={state.opacity}
                        onChange={(e) => onOpacityChange(region.name, parseFloat(e.target.value))}
                        className="flex-1 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-white"
                      />
                      <span className="text-xs text-gray-500 w-8 text-right">
                        {Math.round(state.opacity * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Info Footer - only show atlas warning for intensity data where atlas failed */}
          {!atlasRegistered && inputType === "intensity" && (
            <div className="px-4 py-2 bg-yellow-500/10 border-t border-yellow-500/20">
              <p className="text-xs text-yellow-300/80">
                Atlas registration failed. Download atlas files or check MRI orientation.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
