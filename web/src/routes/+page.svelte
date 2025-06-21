<script lang="ts">
  import { onMount } from 'svelte';
  import ModelStatus from '$lib/components/ModelStatus.svelte';
  import SystemCard from '$lib/components/SystemCard.svelte';
  import TrainForm from '$lib/components/TrainForm.svelte';
  import TrainProgress from '$lib/components/TrainProgress.svelte';
  import { fetchCpuInfo, fetchGpuInfo, fetchModel, fetchTrainStatus, startTrain } from '$lib/api';
  import type { CpuInfo, GpuInfo, ModelInfo, TrainRequest, TrainStatus } from '$lib/api';

  let cpu: CpuInfo | null = null;
  let gpu: GpuInfo | null = null;
  let model: ModelInfo | null = null;
  let trainStatus: TrainStatus | null = null;
  let jobId = '';

  onMount(async () => {
    [cpu, gpu, model] = await Promise.all([fetchCpuInfo(), fetchGpuInfo(), fetchModel()]);
  });

  async function start(request: TrainRequest) {
    const job = await startTrain(request);
    jobId = job.job_id;
    trainStatus = await fetchTrainStatus(jobId);
  }
</script>

<div class="grid">
  <SystemCard
    title="CPU"
    rows={[
      ['Name', cpu?.name],
      ['Physical cores', cpu?.physical_cores],
      ['Logical cores', cpu?.logical_cores],
      ['Usage', cpu ? `${cpu.usage_percent.toFixed(1)}%` : null]
    ]}
  />
  <SystemCard
    title="GPU"
    rows={[
      ['Available', gpu?.available ? 'yes' : 'no'],
      ['Name', gpu?.name],
      ['VRAM', gpu?.vram_mb ? `${gpu.vram_mb} MB` : null],
      ['Driver', gpu?.driver_version]
    ]}
  />
  <ModelStatus {model} />
</div>

<div class="grid lower">
  <TrainForm backend={model?.backend ?? 'cpu'} onStart={start} />
  <TrainProgress status={trainStatus} />
</div>

<style>
  .lower {
    margin-top: 16px;
  }
</style>
