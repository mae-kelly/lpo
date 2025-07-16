"use client"

import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AnalysisResult } from "../types"
import { formatPercent, formatCurrency, formatMultiple } from "@/lib/utils"
import { TrendingUp, AlertTriangle, Target, BarChart3, PieChart } from "lucide-react"
import { SensitivityChart } from "./sensitivity-chart"

interface ResultsDisplayProps {
  result: AnalysisResult
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const { irr_prediction, uncertainty, confidence_interval, risk_factors, recommendation } = result

  const getRecommendationColor = (rec: string) => {
    if (rec.includes('STRONG BUY')) return 'bg-green-600'
    if (rec.includes('BUY')) return 'bg-blue-600'
    if (rec.includes('HOLD')) return 'bg-yellow-600'
    if (rec.includes('PASS')) return 'bg-red-600'
    return 'bg-gray-600'
  }

  const moic = Math.pow(1 + irr_prediction, result.deal_input.hold_period)

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <Card className="glass-effect border-gray-800 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Investment Analysis</h2>
              <p className="text-gray-300">Quantitative assessment and risk evaluation</p>
            </div>
            <Badge className={`${getRecommendationColor(recommendation)} text-white px-4 py-2`}>
              {recommendation}
            </Badge>
          </div>
        </div>
        
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="text-center p-4 rounded-lg bg-gradient-to-br from-blue-500/10 to-blue-600/10 border border-blue-500/20">
              <TrendingUp className="h-8 w-8 text-blue-400 mx-auto mb-2" />
              <div className="text-3xl font-bold text-white mb-1">
                {formatPercent(irr_prediction)}
              </div>
              <div className="text-sm text-gray-400">Projected IRR</div>
              <div className="text-xs text-gray-500 mt-1">
                ±{formatPercent(uncertainty)}
              </div>
            </div>

            <div className="text-center p-4 rounded-lg bg-gradient-to-br from-green-500/10 to-green-600/10 border border-green-500/20">
              <Target className="h-8 w-8 text-green-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white mb-1">
                {moic.toFixed(1)}x
              </div>
              <div className="text-sm text-gray-400">Money Multiple</div>
              <div className="text-xs text-gray-500 mt-1">
                {result.deal_input.hold_period}yr hold
              </div>
            </div>

            <div className="text-center p-4 rounded-lg bg-gradient-to-br from-purple-500/10 to-purple-600/10 border border-purple-500/20">
              <BarChart3 className="h-8 w-8 text-purple-400 mx-auto mb-2" />
              <div className="text-lg font-bold text-white mb-1">
                {formatPercent(confidence_interval[0])} - {formatPercent(confidence_interval[1])}
              </div>
              <div className="text-sm text-gray-400">95% Confidence</div>
            </div>

            <div className="text-center p-4 rounded-lg bg-gradient-to-br from-orange-500/10 to-orange-600/10 border border-orange-500/20">
              <PieChart className="h-8 w-8 text-orange-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white mb-1">
                {formatMultiple(result.deal_input.entry_multiple)}
              </div>
              <div className="text-sm text-gray-400">Entry Multiple</div>
              <div className="text-xs text-gray-500 mt-1">
                {formatPercent(result.deal_input.leverage_ratio)} debt
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-900/30 rounded-lg border border-gray-700">
            <div>
              <div className="text-sm text-gray-400">Revenue</div>
              <div className="font-semibold text-white">{formatCurrency(result.deal_input.ttm_revenue)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">EBITDA</div>
              <div className="font-semibold text-white">{formatCurrency(result.deal_input.ttm_ebitda)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Growth Rate</div>
              <div className="font-semibold text-white">{formatPercent(result.deal_input.revenue_growth)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">EBITDA Margin</div>
              <div className="font-semibold text-white">{formatPercent(result.deal_input.ebitda_margin)}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {risk_factors.length > 0 && (
        <Card className="glass-effect border-gray-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              Risk Assessment
            </CardTitle>
            <CardDescription>
              Key factors that may impact investment performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              {risk_factors.map((risk, index) => (
                <div key={index} className="flex items-start gap-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 flex-shrink-0" />
                  <span className="text-gray-300 text-sm">{risk}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <SensitivityChart dealInput={result.deal_input} />
    </motion.div>
  )
}
