'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  AlertTriangle,
  MapPin,
  Wrench,
  BarChart3,
  Cpu,
  Layers,
  ShieldCheck,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { APP_NAME } from '@/lib/constants';

const navItems = [
  { label: 'Overview', href: '/', icon: LayoutDashboard },
  { label: 'Incidents', href: '/incidents', icon: AlertTriangle },
  { label: 'Road Damage Map', href: '/map', icon: MapPin },
  { label: 'Maintenance Tasks', href: '/maintenance', icon: Wrench },
  { label: 'Analytics & Trends', href: '/analytics', icon: BarChart3 },
  { label: 'ML Models & Registry', href: '/models', icon: Cpu },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-30 flex w-64 flex-col border-r border-slate-800 bg-slate-950/90 backdrop-blur-md">
      {/* Brand */}
      <div className="flex h-16 items-center gap-3 border-b border-slate-800 px-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
          <ShieldCheck className="h-5 w-5" />
        </div>
        <div>
          <span className="text-base font-bold tracking-tight text-white">{APP_NAME}</span>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
            Enterprise AI Ops
          </p>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mb-2 px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Main Console
        </div>
        <nav className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm'
                    : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                )}
              >
                <Icon className={cn('h-4 w-4', isActive ? 'text-emerald-400' : 'text-slate-400')} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-8 mb-2 px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
          Infrastructure
        </div>
        <div className="rounded-lg border border-slate-800/80 bg-slate-900/40 p-3 text-xs">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              API Service
            </span>
            <span className="text-[10px] font-mono text-emerald-400">v0.1.0</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              ML Service
            </span>
            <span className="text-[10px] font-mono text-emerald-400">v0.1.0</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-slate-800 p-4">
        <div className="flex items-center gap-3 rounded-lg bg-slate-900/50 p-2.5 border border-slate-800/60">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-xs font-semibold text-emerald-400">
            AD
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium text-white">Administrator</p>
            <p className="truncate text-[10px] text-slate-400">Municipal Operations</p>
          </div>
        </div>
      </div>
    </aside>
  );
};
