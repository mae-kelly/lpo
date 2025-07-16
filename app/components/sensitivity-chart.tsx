"use client"

import { useState, useEffect } from "react"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"
import { Line } from "react-chartjs-2"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { DealInput } from "../types"
import { formatPercent } from "@/lib/utils"
import { Activity } from "lucide-react"

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface SensitivityChartProps {
  dealInput: DealInput
}

export function SensitivityChart({ dealInput }: SensitivityChartProps) {
  const [chartData, setChartData] = useState<any>(null)

  useEffect(() => {
    const generateSensitivityData = () => {
      const exitMultiples = []
      const irrData = []
      
      for (let multiple = 6; multiple <= 20; multiple += 0.5) {
        exitMultiples.push(multiple.toFixed(1))
        
        const enterpriseValue = dealInput.ttm_ebitda * dealInput.entry_multiple
        const exitValue = dealInput.ttm_ebitda * multiple * Math.pow(1 + dealInput.revenue_growth, dealInput.hold_period)
        const debtPaydown = enterpriseValue * dealInput.leverage_ratio * 0.7
        const netProceeds = exitValue - debtPaydown
        const initialEquity = enterpriseValue * (1 - dealInput.leverage_ratio)
        const moic = Math.max(0, netProceeds / initialEquity)
        const irr = Math.max(0, Math.min(Math.pow(moic, 1 / dealInput.hold_period) - 1, 0.8))
        
        irrData.push(irr)
      }

      return {
        labels: exitMultiples,
        datasets: [
          {
            label: 'Projected IRR',
            data: irrData,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 2,
            pointHoverRadius: 6,
          },
          {
            label: '20% IRR Threshold',
            data: new Array(exitMultiples.length).fill(0.20),
            borderColor: 'rgb(34, 197, 94)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
          },
          {
            label: '15% IRR Minimum',
            data: new Array(exitMultiples.length).fill(0.15),
            borderColor: 'rgb(239, 68, 68)',
            borderDash: [3, 3],
            fill: false,
            pointRadius: 0,
          },
        ],
      }
    }

    setChartData(generateSensitivityData())
  }, [dealInput])

  const options = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: 'rgb(156, 163, 175)',
          usePointStyle: true,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'rgb(255, 255, 255)',
        bodyColor: 'rgb(156, 163, 175)',
        borderColor: 'rgb(75, 85, 99)',
        borderWidth: 1,
        callbacks: {
          label: function(context: any) {
            const value = context.parsed.y
            if (context.datasetIndex === 0) {
              return `${context.dataset.label}: ${formatPercent(value)}`
            }
            return `${context.dataset.label}: ${formatPercent(value)}`
          },
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Exit Multiple (x EBITDA)',
          color: 'rgb(156, 163, 175)',
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Internal Rate of Return',
          color: 'rgb(156, 163, 175)',
        },
        grid: {
          color: 'rgba(75, 85, 99, 0.3)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
          callback: function(value: any) {
            return formatPercent(value)
          },
        },
        min: 0,
        max: 0.6,
      },
    },
  }

  if (!chartData) {
    return (
      <Card className="glass-effect border-gray-800">
        <CardContent className="p-6">
          <div className="h-64 bg-gray-900/30 animate-pulse rounded-lg flex items-center justify-center">
            <div className="text-gray-500">Loading sensitivity analysis...</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="glass-effect border-gray-800">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Activity className="h-5 w-5 text-green-400" />
          IRR Sensitivity Analysis
        </CardTitle>
        <CardDescription>
          Return sensitivity to exit multiple assumptions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <Line data={chartData} options={options} />
        </div>
      </CardContent>
    </Card>
  )
}
