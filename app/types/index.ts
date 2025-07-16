export interface DealInput {
  industry: string;
  ttm_revenue: number;
  ttm_ebitda: number;
  revenue_growth: number;
  ebitda_margin: number;
  entry_multiple: number;
  leverage_ratio: number;
  hold_period: number;
}

export interface ModelResponse {
  irr_prediction: number;
  uncertainty: number;
  confidence_interval: [number, number];
  risk_factors: string[];
  recommendation: string;
}

export interface AnalysisResult extends ModelResponse {
  deal_input: DealInput;
  timestamp: string;
}
