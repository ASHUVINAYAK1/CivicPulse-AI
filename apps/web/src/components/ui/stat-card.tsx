import React from 'react';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
    period: string;
  };
  variant?: 'default' | 'emerald' | 'amber' | 'rose' | 'indigo';
  isPlaceholder?: boolean;
}

const variantStyles = {
  default: 'border-slate-800 bg-slate-900/60 text-slate-100',
  emerald: 'border-emerald-500/30 bg-emerald-950/20 text-slate-100',
  amber: 'border-amber-500/30 bg-amber-950/20 text-slate-100',
  rose: 'border-rose-500/30 bg-rose-950/20 text-slate-100',
  indigo: 'border-indigo-500/30 bg-indigo-950/20 text-slate-100',
};

const iconStyles = {
  default: 'bg-slate-800 text-slate-300',
  emerald: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
  amber: 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
  rose: 'bg-rose-500/10 text-rose-400 border border-rose-500/20',
  indigo: 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20',
};

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  variant = 'default',
  isPlaceholder = false,
}) => {
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-xl border p-5 backdrop-blur-sm transition-all hover:border-slate-700',
        variantStyles[variant]
      )}
    >
      {isPlaceholder && (
        <div className="absolute right-3 top-3">
          <span className="inline-flex items-center rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400 border border-slate-700/60">
            Placeholder
          </span>
        </div>
      )}

      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</p>
        <div className={cn('rounded-lg p-2.5', iconStyles[variant])}>
          <Icon className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-4">
        <h3 className="text-2xl font-bold tracking-tight text-white">{value}</h3>
        {subtitle && <p className="mt-1 text-xs text-slate-400">{subtitle}</p>}
      </div>

      {trend && (
        <div className="mt-4 flex items-center gap-1.5 text-xs">
          <span
            className={cn(
              'font-semibold',
              trend.isPositive ? 'text-emerald-400' : 'text-rose-400'
            )}
          >
            {trend.isPositive ? '+' : '-'}{Math.abs(trend.value)}%
          </span>
          <span className="text-slate-500">{trend.period}</span>
        </div>
      )}
    </div>
  );
};
