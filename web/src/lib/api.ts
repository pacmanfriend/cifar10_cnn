const API_BASE = import.meta.env.PUBLIC_API_BASE ?? '';

export type Backend = 'cpu' | 'gpu';

export type CpuInfo = {
  name: string;
  physical_cores: number | null;
  logical_cores: number;
  usage_percent: number;
};

export type GpuInfo = {
  available: boolean;
  name: string | null;
  vram_mb: number | null;
  driver_version: string | null;
  error: string | null;
};

export type ModelInfo = {
  architecture: string;
  param_count: number;
  backend: Backend;
  loaded: boolean;
};

export type TrainRequest = {
  data_dir: string;
  backend?: Backend;
  epochs?: number;
  learning_rate?: number;
  lr_decay_epochs?: number;
  lr_decay_factor?: number;
  batch_size?: number;
  momentum?: number;
};

export type TrainStart = {
  job_id: string;
};

export type TrainStatus = {
  status: 'running' | 'done' | 'error' | string;
  epoch: number;
  loss: number | null;
  accuracy: number | null;
  error: string | null;
};

export type PredictResponse = {
  class_id: number;
  class_name: string;
  confidence: number;
  scores: number[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(body.error ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

export function fetchCpuInfo() {
  return request<CpuInfo>('/api/system/cpu');
}

export function fetchGpuInfo() {
  return request<GpuInfo>('/api/system/gpu');
}

export function fetchModel() {
  return request<ModelInfo>('/api/model');
}

export function setBackend(backend: Backend) {
  return request<ModelInfo>('/api/model/backend', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ backend })
  });
}

export function loadWeights(path: string) {
  return request('/api/model/load', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path })
  });
}

export function saveWeights(path: string) {
  return request('/api/model/save', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path })
  });
}

export function startTrain(requestBody: TrainRequest) {
  return request<TrainStart>('/api/train', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(requestBody)
  });
}

export function fetchTrainStatus(jobId: string) {
  return request<TrainStatus>(`/api/train/${jobId}`);
}

export function predictImage(file: File) {
  const form = new FormData();
  form.append('image', file);
  return request<PredictResponse>('/api/predict', {
    method: 'POST',
    body: form
  });
}
