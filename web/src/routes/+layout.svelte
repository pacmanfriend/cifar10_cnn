<script lang="ts">
  const links = [
    ['/', 'Dashboard'],
    ['/system', 'System'],
    ['/model', 'Model'],
    ['/train', 'Train'],
    ['/predict', 'Predict']
  ];
</script>

<svelte:head>
  <title>CIFAR-10 CNN</title>
</svelte:head>

<div class="shell">
  <aside>
    <div class="brand">CIFAR-10 CNN</div>
    <nav>
      {#each links as [href, label]}
        <a {href}>{label}</a>
      {/each}
    </nav>
  </aside>
  <main>
    <slot />
  </main>
</div>

<style>
  :global(*) {
    box-sizing: border-box;
  }

  :global(body) {
    margin: 0;
    background: #f4f6f8;
    color: #17202a;
    font-family:
      Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  :global(button),
  :global(input),
  :global(select) {
    font: inherit;
  }

  .shell {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr);
    min-height: 100vh;
  }

  aside {
    background: #101820;
    color: #f7fafc;
    padding: 22px 18px;
  }

  .brand {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 28px;
  }

  nav {
    display: grid;
    gap: 6px;
  }

  a {
    color: #d7dee7;
    text-decoration: none;
    padding: 10px 12px;
    border-radius: 6px;
  }

  a:hover {
    background: #22313f;
    color: white;
  }

  main {
    padding: 28px;
    min-width: 0;
  }

  :global(.grid) {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
  }

  :global(.panel) {
    background: white;
    border: 1px solid #dce3ea;
    border-radius: 8px;
    padding: 18px;
  }

  :global(.panel h2) {
    margin: 0 0 14px;
    font-size: 18px;
  }

  :global(.metric) {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    padding: 8px 0;
    border-bottom: 1px solid #edf1f5;
  }

  :global(.metric:last-child) {
    border-bottom: 0;
  }

  :global(.actions) {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }

  :global(button) {
    border: 0;
    border-radius: 6px;
    background: #146c94;
    color: white;
    padding: 9px 13px;
    cursor: pointer;
  }

  :global(button.secondary) {
    background: #52616b;
  }

  :global(input),
  :global(select) {
    width: 100%;
    border: 1px solid #cdd7e1;
    border-radius: 6px;
    padding: 9px 10px;
  }

  :global(label) {
    display: grid;
    gap: 6px;
    color: #334155;
    font-size: 14px;
  }

  @media (max-width: 760px) {
    .shell {
      grid-template-columns: 1fr;
    }

    aside {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 14px;
    }

    nav {
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }

    a {
      text-align: center;
      padding: 8px 4px;
      font-size: 13px;
    }

    main {
      padding: 18px;
    }
  }
</style>
