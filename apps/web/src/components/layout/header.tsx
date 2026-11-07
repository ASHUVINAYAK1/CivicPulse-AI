'use client';

import React from 'react';
import { Bell, Search, Activity, ShieldAlert, Sparkles } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="sticky top-0 z-20 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-slate-950/80 px-8 backdrop-blur-md">
      {/* Search Bar */}
      <div className="relative w-96">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          placeholder="Search incidents by ID, road, or severity..."
          className="w-full rounded-lg border border-slate-800 bg-slate-900/60 py-1.5 pl-9 pr-4 text-xs text-slate-200 placeholder-slate-500 focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
        />
      </div>

      {/* Actions */}
      <div className="flex items-center gap-4">
        {/* Environment Indicator */}
        <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-950/30 px-3 py-1 text-xs font-medium text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>Local Environment</span>
        </div>

        {/* AI Engine Status */}
        <div className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-300">
          <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
          <span className="font-mono text-[11px] text-slate-400">ML Engine Ready</span>
        </div>

        {/* Notifications */}
        <button
          type="button"
          className="relative rounded-lg border border-slate-800 bg-slate-900/60 p-2 text-slate-400 hover:text-white hover:border-slate-700 transition-colors"
          aria-label="Notifications"
        >
          <Bell className="h-4 w-4" />
          <span className="absolute -right-0.5 -top-0.5 flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
          </span>
        </button>
      </div>
    </header>
  );
};
