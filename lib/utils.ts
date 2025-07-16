import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

export const formatPercent = (value: number, decimals = 1): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatMultiple = (value: number): string => {
  return `${value.toFixed(1)}x`;
};

export const industries = [
  'Technology',
  'Healthcare Services',
  'Industrial Services',
  'Consumer Discretionary',
  'Consumer Staples',
  'Energy Services',
  'Financial Services',
  'Business Services',
  'Real Estate',
  'Communications',
  'Infrastructure'
];
