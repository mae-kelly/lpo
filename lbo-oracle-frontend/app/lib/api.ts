import axios from 'axios';
import { DealInput, ModelResponse } from '../types';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const analyzeDeal = async (dealInput: DealInput): Promise<ModelResponse> => {
  try {
    const response = await api.post<ModelResponse>('/predict', dealInput);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || 'Analysis failed');
    }
    throw error;
  }
};

export const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await api.get('/health');
    return response.data.status === 'healthy';
  } catch {
    return false;
  }
};
