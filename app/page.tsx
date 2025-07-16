"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { DealForm } from "./components/deal-form"
import { ResultsDisplay } from "./components/results-display"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { DealInput, AnalysisResult } from "./types"
import { analyzeDeal, healthCheck } from "./lib/api"
import { Wallet, Activity, Globe, Settings, ChevronDown, Search } from "lucide-react"
import { Input } from "@/components/ui/input"

export default function Page() {
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiHealth, setApiHealth] = useState<boolean | null>(null)

  useEffect(() => {
    const checkApi = async () => {
      const health = await healthCheck()
      setApiHealth(health)
    }
    checkApi()
  }, [])

  const handleDealSubmit = async (dealInput: DealInput) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await analyzeDeal(dealInput)
      setResult({
        ...response,
        deal_input: dealInput,
        timestamp: new Date().toISOString(),
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="grid lg:grid-cols-[280px_1fr]">
        <aside className="border-r border-gray-800 bg-gray-900/50 backdrop-blur">
          <div className="flex h-16 items-center gap-2 border-b border-gray-800 px-6">
            <Wallet className="h-6 w-6 text-blue-400" />
            <span className="font-bold gradient-text text-xl">LBO-ORACLE™</span>
          </div>
          
          <div className="px-4 py-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input 
                placeholder="Search deals..." 
                className="pl-10 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400"
              />
            </div>
          </div>
          
          <nav className="space-y-2 px-2">
            <Button variant="ghost" className="w-full justify-start gap-2 text-white hover:bg-gray-800">
              <Activity className="h-4 w-4" />
              Deal Analysis
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-gray-400 hover:text-white hover:bg-gray-800">
              <Globe className="h-4 w-4" />
              Market Intelligence
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-gray-400 hover:text-white hover:bg-gray-800">
              <Wallet className="h-4 w-4" />
              Portfolio Tracker
              <ChevronDown className="ml-auto h-4 w-4" />
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-2 text-gray-400 hover:text-white hover:bg-gray-800">
              <Settings className="h-4 w-4" />
              Configuration
            </Button>
          </nav>

          <div className="absolute bottom-4 left-4 right-4">
            <div className={`flex items-center gap-2 text-xs ${apiHealth ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full ${apiHealth ? 'bg-green-400' : 'bg-red-400'}`} />
              {apiHealth ? 'API Connected' : 'API Offline'}
            </div>
          </div>
        </aside>

        <main className="p-6">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 flex items-center justify-between"
          >
            <div className="space-y-1">
              <h1 className="text-3xl font-bold gradient-text">Investment Analysis</h1>
              <p className="text-gray-400">Advanced LBO modeling and risk assessment</p>
            </div>
            <Button variant="outline" className="gap-2 border-gray-700 text-gray-300 hover:text-white">
              Neural Network
              <ChevronDown className="h-4 w-4" />
            </Button>
          </motion.div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div>
              <DealForm onSubmit={handleDealSubmit} loading={loading} />
              
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4"
                >
                  <Card className="border-red-500/50 bg-red-500/10">
                    <CardContent className="p-4">
                      <div className="text-red-400 text-sm flex items-center gap-2">
                        <div className="w-2 h-2 bg-red-400 rounded-full" />
                        {error}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </div>
            
            <div>
              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <Card className="glass-effect border-gray-800">
                    <CardContent className="p-8">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-400" />
                          <span className="text-gray-300">Processing financial model...</span>
                        </div>
                        <div className="space-y-3">
                          <div className="h-4 bg-gray-700 rounded animate-pulse" />
                          <div className="h-4 bg-gray-700 rounded animate-pulse w-3/4" />
                          <div className="h-4 bg-gray-700 rounded animate-pulse w-1/2" />
                        </div>
                        <div className="text-xs text-gray-500">
                          Running neural network inference and risk assessment...
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
              
              {result && !loading && (
                <ResultsDisplay result={result} />
              )}
              
              {!result && !loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <Card className="glass-effect border-gray-800">
                    <CardContent className="p-8 text-center">
                      <div className="space-y-4">
                        <div className="w-16 h-16 mx-auto bg-gray-800 rounded-full flex items-center justify-center">
                          <Activity className="h-8 w-8 text-gray-500" />
                        </div>
                        <div>
                          <h3 className="text-lg font-medium text-gray-300 mb-2">
                            Ready for Analysis
                          </h3>
                          <p className="text-gray-500 text-sm">
                            Configure deal parameters and leverage our neural network 
                            to generate sophisticated investment analysis
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </div>
          </div>
        </main>
