"use client";

import { useState } from "react";

interface CreateCaseModalProps {
  isOpen: boolean;
  patientId: number;
  onCreateCase: (caseName: string) => void;
  onCancel: () => void;
  isCreating?: boolean;
}

export default function CreateCaseModal({
  isOpen,
  patientId,
  onCreateCase,
  onCancel,
  isCreating = false,
}: CreateCaseModalProps) {
  const [caseName, setCaseName] = useState("");

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (caseName.trim()) {
      onCreateCase(caseName.trim());
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onCancel}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
        <h3 className="text-xl font-bold text-gray-900 mb-4">
          Create New Case
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Patient ID: <span className="font-semibold">{patientId}</span>
        </p>

        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label
              htmlFor="case-name"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Case Name
            </label>
            <input
              id="case-name"
              type="text"
              value={caseName}
              onChange={(e) => setCaseName(e.target.value)}
              placeholder="Enter case name..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isCreating}
              autoFocus
            />
          </div>

          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={onCancel}
              disabled={isCreating}
              className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!caseName.trim() || isCreating}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
            >
              {isCreating ? "Creating..." : "Create Case"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
