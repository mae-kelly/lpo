'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { motion } from 'framer-motion';
import { DealInput } from '../types';
import { industries } from '../lib/utils';

interface DealFormProps {
  onSubmit: (data: DealInput) => void;
  loading: boolean;
}

export default function DealForm({ onSubmit, loading }: DealFormProps) {
  const { register, handleSubmit, formState: { errors }, watch } = useForm<DealInput>({
    defaultValues: {
      industry: 'Technology',
      ttm_revenue: 100000000,
      ttm_ebitda: 15000000,
      revenue_growth: 0.05,
      ebitda_margin: 0.15,
      entry_multiple: 10.0,
      leverage_ratio: 0.60,
      hold_period: 5,
    }
  });

  const watchedValues = watch();

  const enterpriseValue = watchedValues.ttm_ebitda * watchedValues.entry_multiple;
  const debtAmount = enterpriseValue * watchedValues.leverage_ratio;
  const equityAmount = enterpriseValue - debtAmount;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white shadow-lg rounded-lg p-8"
    >
      <h2 className="text-2xl font-serif text-navy-800 mb-6">Deal Parameters</h2>
      
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Industry
            </label>
            <select
              {...register('industry', { required: 'Industry is required' })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            >
              {industries.map(industry => (
                <option key={industry} value={industry}>{industry}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              TTM Revenue ($)
            </label>
            <input
              type="number"
              {...register('ttm_revenue', { 
                required: 'Revenue is required',
                min: { value: 1000000, message: 'Minimum $1M revenue' }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
            {errors.ttm_revenue && (
              <p className="text-red-500 text-sm mt-1">{errors.ttm_revenue.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              TTM EBITDA ($)
            </label>
            <input
              type="number"
              {...register('ttm_ebitda', { 
                required: 'EBITDA is required',
                min: { value: 100000, message: 'Minimum $100K EBITDA' }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
            {errors.ttm_ebitda && (
              <p className="text-red-500 text-sm mt-1">{errors.ttm_ebitda.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Revenue Growth (Annual %)
            </label>
            <input
              type="number"
              step="0.01"
              {...register('revenue_growth', { 
                required: 'Growth rate is required',
                min: { value: -0.1, message: 'Minimum -10%' },
                max: { value: 0.5, message: 'Maximum 50%' }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              EBITDA Margin (%)
            </label>
            <input
              type="number"
              step="0.01"
              {...register('ebitda_margin', { 
                required: 'EBITDA margin is required',
                min: { value: 0.01, message: 'Minimum 1%' },
                max: { value: 0.5, message: 'Maximum 50%' }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Entry Multiple (x EBITDA)
            </label>
            <input
              type="number"
              step="0.1"
              {...register('entry_multiple', { 
                required: 'Entry multiple is required',
                min: { value: 3, message: 'Minimum 3.0x' },
                max: { value: 25, message: 'Maximum 25.0x' }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Leverage Ratio (%)
            </label>
            <input
              type="number"
              step="0.01"
              {...register('leverage_ratio', { 
                required: 'Leverage ratio is required',
                min: { value: 0.1, message: 'Minimum 10%' },
                max: { value: 0.8, message: 'Maximum 80%' }
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Hold Period (Years)
            </label>
            <select
              {...register('hold_period', { required: 'Hold period is required' })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-navy-500"
            >
              {[3, 4, 5, 6, 7].map(years => (
                <option key={years} value={years}>{years} years</option>
              ))}
            </select>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="text-lg font-medium text-gray-900 mb-3">Transaction Summary</h3>
          <div className="
