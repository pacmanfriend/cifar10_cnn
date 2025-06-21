<script lang="ts">
  export let title: string;
  export let rows: Array<[string, string | number | null | undefined]> = [];
</script>

<section class="panel">
  <h2>{title}</h2>
  {#each rows as [label, value]}
    <div class="metric">
      <span>{label}</span>
      <strong>{value ?? 'n/a'}</strong>
    </div>
  {/each}
</section>
