'use strict';
window.FusionCharts = (() => {
  const ns = 'http://www.w3.org/2000/svg';
  function node(tag, attributes = {}, text) {
    const el = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([k, v]) => el.setAttribute(k, v));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  const fmt = n => Number(n).toLocaleString('zh-CN', {maximumFractionDigits: 3});
  function traces(container, stream, duration, color, queryTime) {
    container.replaceChildren();
    stream.traces.forEach(trace => {
      const fig = document.createElement('figure');
      const caption = document.createElement('figcaption');
      caption.textContent = trace.label;
      fig.append(caption);
      const width = Math.max(240, container.clientWidth), left = 78, right = width - 8;
      const svg = node('svg', {viewBox: `0 0 ${width} 110`, role: 'img', 'aria-label': trace.label + '随时间变化'});
      const values = trace.points.map(p => p[1]);
      let low = Math.min(...values), high = Math.max(...values);
      if (low === high) { low -= .5; high += .5; }
      [15, 75].forEach(y => svg.append(node('line', {x1: left, x2: right, y1: y, y2: y, stroke: '#dce5e5', 'stroke-dasharray': '3 4'})));
      const d = trace.points.map(([t,v], i) => `${i ? 'L' : 'M'}${left + t / duration * (right-left)},${75-(v-low)/(high-low)*60}`).join(' ');
      svg.append(node('path', {d, fill: 'none', stroke: color, 'stroke-width': 1.6}));
      const x = left + queryTime / duration * (right-left);
      svg.append(node('line', {x1:x, x2:x, y1:9, y2:80, stroke:'#18343e', 'stroke-dasharray':'3 3'}));
      [[0,18,fmt(high)],[0,78,fmt(low)],[left,103,'0'],[right,103,fmt(duration)+' 秒']].forEach(([x,y,t], i) => {
        svg.append(node('text',{x,y, ...(i===3 ? {'text-anchor':'end'} : {})},t));
      });
      const before = trace.points.filter(p => p[0] <= queryTime).at(-1);
      const value = document.createElement('div'); value.className='trace-value';
      value.textContent = before ? `${fmt(before[0])} 秒 · ${fmt(before[1])}` : '该时刻之前暂无观测';
      fig.append(svg,value); container.append(fig);
    });
    if (!stream.traces.length) container.textContent='该窗口暂无有效观测。';
  }
  function heatmap(canvas, w, scale, q, dimension) {
    const ctx=canvas.getContext('2d'), width=canvas.width, height=canvas.height;
    const nt=w.embedding.length, nf=w.embedding[0].length;
    ctx.fillStyle='#e4e9e8'; ctx.fillRect(0,0,width,height);
    for (let t=0;t<nt;t++) for(let f=0;f<nf;f++) {
      if(!w.valid[t]) continue;
      const v=w.embedding[t][f]/scale, base=[242,243,235], target=v<0?[33,123,139]:[217,155,80];
      ctx.fillStyle=`rgb(${base.map((b,i)=>Math.round(b+(target[i]-b)*Math.abs(v))).join(',')})`;
      ctx.fillRect(t*width/nt,f*height/nf,width/nt+1,height/nf+1);
    }
    ctx.strokeStyle='#163a46'; ctx.lineWidth=2;
    ctx.strokeRect(q*width/nt,1,width/nt,height-2);
    ctx.strokeRect(1,dimension*height/nf,width-2,height/nf);
  }
  return {traces,heatmap};
})();
