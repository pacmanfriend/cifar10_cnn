<script lang="ts">
  import { onMount } from 'svelte';
  import SystemCard from '$lib/components/SystemCard.svelte';
  import { fetchCpuInfo, fetchGpuInfo } from '$lib/api';
  import type { CpuInfo, GpuInfo } from '$lib/api';

  let cpu: CpuInfo | null = null;
  let gpu: GpuInfo | null = null;

  onMount(async () => {
    [cpu, gpu] = await Promise.all([fetchCpuInfo(), fetchGpuInfo()]);
  });
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
      ['Driver', gpu?.driver_version],
      ['Error', gpu?.error]
    ]}
  />
</div>
