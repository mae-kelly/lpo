import { clsx, type ClassValue } from 'clsx';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
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

export const getRecommendationColor = (recommendation: string): string => {
  if (recommendation.includes('STRONG BUY')) return 'text-green-600 bg-green-50';
  if (recommendation.includes('BUY')) return 'text-blue-600 bg-blue-50';
  if (recommendation.includes('HOLD')) return 'text-yellow-600 bg-yellow-50';
  if (recommendation.includes('PASS')) return 'text-red-600 bg-red-50';
  return 'text-gray-600 bg-gray-50';
};

export const industries = [
  'Technology',
  'Healthcare',
  'Industrials',
  'Consumer Discretionary',
  'Consumer Staples',
  'Energy',
  'Financials',
  'Materials',
  'Real Estate',
  'Communication Services',
  'Utilities'
];
