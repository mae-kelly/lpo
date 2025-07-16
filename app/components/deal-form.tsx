"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DealInput } from "../types"
import { industries, formatCurrency } from "@/lib/utils"
import { Calculator, TrendingUp, DollarSign } from "lucide-react"

interface DealFormProps {
  onSubmit: (data: DealInput) => void
  loading: boolean
}

export function DealForm({ onSubmit, loading }: DealFormProps) {
  const { register, handleSubmit, formState: { errors }, watch, setValue } = useForm<DealInput>({
    defaultValues: {
      industry: 'Technology',
      ttm_revenue: 100000000,
      ttm_ebitda: 20000000,
      revenue_growth: 0.08,
      ebitda_margin: 0.20,
      entry_multiple: 12.0,
      leverage_ratio: 0.50,
      hold_period: 5,
    }
  })

  const watchedValues = watch()
  const enterpriseValue = watchedValues.ttm_ebitda * watchedValues.entry_multiple
  const debtAmount = enterpriseValue * watchedValues.leverage_ratio
  const equityAmount = enterpriseValue - debtAmount

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="glass-effect border-gray-800">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Calculator className="h-5 w-5 text-blue-400" />
            Deal Parameters
          </CardTitle>
          <CardDescription>
            Configure the investment structure and financial assumptions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              <div className="space-y-2">
                <Label htmlFor="industry">Industry Sector</Label>
                <Select
                  value={watchedValues.industry}
                  onValueChange={(value) => setValue('industry', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select industry" />
                  </SelectTrigger>
                  <SelectContent>
                    {industries.map(industry => (
                      <SelectItem key={industry} value={industry}>
                        {industry}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="hold_period">Investment Horizon</Label>
                <Select
                  value={watchedValues.hold_period.toString()}
                  onValueChange={(value) => setValue('hold_period', parseInt(value))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[3, 4, 5, 6, 7].map(years => (
                      <SelectItem key={years} value={years.toString()}>
                        {years} years
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="ttm_revenue">TTM Revenue</Label>
                <Input
                  id="ttm_revenue"
                  type="number"
                  {...register('ttm_revenue', { 
                    required: 'Revenue is required',
                    min: { value: 10000000, message: 'Minimum $10M revenue' }
                  })}
                  className="bg-gray-900/50 border-gray-700"
                />
                {errors.ttm_revenue && (
                  <p className="text-red-400 text-sm">{errors.ttm_revenue.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="ttm_ebitda">TTM EBITDA</Label>
                <Input
                  id="ttm_ebitda"
                  type="number"
                  {...register('ttm_ebitda', { 
                    required: 'EBITDA is required',
                    min: { value: 1000000, message: 'Minimum $1M EBITDA' }
                  })}
                  className="bg-gray-900/50 border-gray-700"
                />
                {errors.ttm_ebitda && (
                  <p className="text-red-400 text-sm">{errors.ttm_ebitda.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="revenue_growth">Revenue Growth (Annual %)</Label>
                <Input
                  id="revenue_growth"
                  type="number"
                  step="0.01"
                  {...register('revenue_growth', { 
                    required: 'Growth rate is required',
                    min: { value: -0.1, message: 'Minimum -10%' },
                    max: { value: 0.5, message: 'Maximum 50%' }
                  })}
                  className="bg-gray-900/50 border-gray-700"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="ebitda_margin">EBITDA Margin (%)</Label>
                <Input
                  id="ebitda_margin"
                  type="number"
                  step="0.01"
                  {...register('ebitda_margin', { 
                    required: 'EBITDA margin is required',
                    min: { value: 0.05, message: 'Minimum 5%' },
                    max: { value: 0.8, message: 'Maximum 80%' }
                  })}
                  className="bg-gray-900/50 border-gray-700"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="entry_multiple">Entry Multiple (x EBITDA)</Label>
                <Input
                  id="entry_multiple"
                  type="number"
                  step="0.1"
                  {...register('entry_multiple', { 
                    required: 'Entry multiple is required',
                    min: { value: 3, message: 'Minimum 3.0x' },
                    max: { value: 30, message: 'Maximum 30.0x' }
                  })}
                  className="bg-gray-900/50 border-gray-700"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="leverage_ratio">Leverage Ratio (%)</Label>
                <Input
                  id="leverage_ratio"
                  type="number"
                  step="0.01"
                  {...register('leverage_ratio', { 
                    required: 'Leverage ratio is required',
                    min: { value: 0.1, message: 'Minimum 10%' },
                    max: { value: 0.85, message: 'Maximum 85%' }
                  })}
                  className="bg-gray-900/50 border-gray-700"
                />
              </div>
            </div>

            <Card className="bg-gray-900/30 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <DollarSign className="h-4 w-4 text-green-400" />
                  Transaction Overview
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">
                      {formatCurrency(enterpriseValue)}
                    </div>
                    <div className="text-sm text-gray-400">Enterprise Value</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-400">
                      {formatCurrency(debtAmount)}
                    </div>
                    <div className="text-sm text-gray-400">Total Debt</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">
                      {formatCurrency(equityAmount)}
                    </div>
                    <div className="text-sm text-gray-400">Equity Investment</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-medium py-3"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Analyzing Deal...
                </>
              ) : (
                <>
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Analyze Investment
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </motion.div>
  )
}
