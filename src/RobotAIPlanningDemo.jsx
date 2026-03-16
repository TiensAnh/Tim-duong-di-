import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw, Shuffle, Bot, Home, Brain, Map, Target, Lightbulb, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

const CELL = {
  EMPTY: 0,
  WALL: 1,
  START: 2,
  GOAL: 3,
  OPEN: 4,
  CLOSED: 5,
  PATH: 6,
  AGENT: 7,
  RANDOM: 8,
};

const DIRS4 = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

const ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"];

const colorMap = {
  [CELL.EMPTY]: "bg-slate-100 dark:bg-slate-800",
  [CELL.WALL]: "bg-slate-900 dark:bg-slate-200",
  [CELL.START]: "bg-emerald-500",
  [CELL.GOAL]: "bg-rose-500",
  [CELL.OPEN]: "bg-sky-300",
  [CELL.CLOSED]: "bg-indigo-300",
  [CELL.PATH]: "bg-amber-400",
  [CELL.AGENT]: "bg-fuchsia-500",
  [CELL.RANDOM]: "bg-orange-400",
};

function keyOf(r, c) {
  return `${r},${c}`;
}

function parseKey(key) {
  return key.split(",").map(Number);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function createGrid(rows, cols, wallProbability = 0.22) {
  const g = Array.from({ length: rows }, () => Array(cols).fill(CELL.EMPTY));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (Math.random() < wallProbability) g[r][c] = CELL.WALL;
    }
  }
  return g;
}

function carveMaze(rows, cols) {
  const g = Array.from({ length: rows }, () => Array(cols).fill(CELL.WALL));

  const inside = (r, c) => r > 0 && r < rows - 1 && c > 0 && c < cols - 1;
  const startR = rows % 2 === 0 ? 1 : 1;
  const startC = cols % 2 === 0 ? 1 : 1;
  g[startR][startC] = CELL.EMPTY;
  const stack = [[startR, startC]];

  const steps = [
    [2, 0],
    [-2, 0],
    [0, 2],
    [0, -2],
  ];

  while (stack.length) {
    const [r, c] = stack[stack.length - 1];
    const neighbors = [];

    for (const [dr, dc] of steps) {
      const nr = r + dr;
      const nc = c + dc;
      if (inside(nr, nc) && g[nr][nc] === CELL.WALL) {
        neighbors.push([nr, nc, r + dr / 2, c + dc / 2]);
      }
    }

    if (!neighbors.length) {
      stack.pop();
      continue;
    }

    const [nr, nc, wr, wc] = neighbors[randomInt(0, neighbors.length - 1)];
    g[wr][wc] = CELL.EMPTY;
    g[nr][nc] = CELL.EMPTY;
    stack.push([nr, nc]);
  }

  for (let r = 1; r < rows - 1; r++) {
    for (let c = 1; c < cols - 1; c++) {
      if (g[r][c] === CELL.WALL && Math.random() < 0.08) g[r][c] = CELL.EMPTY;
    }
  }

  return g;
}

function placeStartGoal(grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  const start = [1, 1];
  const goal = [rows - 2, cols - 2];
  grid[start[0]][start[1]] = CELL.EMPTY;
  grid[goal[0]][goal[1]] = CELL.EMPTY;
  return { grid, start, goal };
}

function copyGrid(grid) {
  return grid.map((row) => [...row]);
}

function heuristic(a, b) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

function validNeighbors(grid, r, c) {
  const rows = grid.length;
  const cols = grid[0].length;
  const out = [];
  for (const [dr, dc] of DIRS4) {
    const nr = r + dr;
    const nc = c + dc;
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] !== CELL.WALL) {
      out.push([nr, nc]);
    }
  }
  return out;
}

function reconstruct(cameFrom, endKey) {
  const path = [];
  let cur = endKey;
  while (cur) {
    path.push(parseKey(cur));
    cur = cameFrom[cur];
  }
  return path.reverse();
}

function genericSearch(grid, start, goal, algorithm = "astar") {
  const startKey = keyOf(...start);
  const goalKey = keyOf(...goal);

  const frontier = [{ key: startKey, pos: start, g: 0, h: heuristic(start, goal), f: heuristic(start, goal) }];
  const visited = new Set();
  const cameFrom = {};
  const gScore = { [startKey]: 0 };
  const steps = [];

  while (frontier.length) {
    frontier.sort((a, b) => a.f - b.f || a.h - b.h || a.g - b.g);
    const current = frontier.shift();
    if (!current) break;
    if (visited.has(current.key)) continue;
    visited.add(current.key);
    steps.push({ type: "close", node: current.pos });

    if (current.key === goalKey) {
      const path = reconstruct(cameFrom, goalKey);
      for (const p of path) steps.push({ type: "path", node: p });
      return { found: true, path, steps, visitedCount: visited.size };
    }

    for (const [nr, nc] of validNeighbors(grid, current.pos[0], current.pos[1])) {
      const nk = keyOf(nr, nc);
      const tentativeG = current.g + 1;

      if (algorithm === "random") {
        if (!visited.has(nk) && !frontier.some((x) => x.key === nk)) {
          cameFrom[nk] = current.key;
          frontier.push({ key: nk, pos: [nr, nc], g: tentativeG, h: Math.random() * 50, f: Math.random() * 50 });
          steps.push({ type: "open", node: [nr, nc] });
        }
        continue;
      }

      if (gScore[nk] === undefined || tentativeG < gScore[nk]) {
        gScore[nk] = tentativeG;
        cameFrom[nk] = current.key;
        const h = heuristic([nr, nc], goal);
        let f = tentativeG + h;
        if (algorithm === "dijkstra") f = tentativeG;
        if (algorithm === "greedy") f = h;
        frontier.push({ key: nk, pos: [nr, nc], g: tentativeG, h, f });
        steps.push({ type: "open", node: [nr, nc] });
      }
    }
  }

  return { found: false, path: [], steps, visitedCount: visited.size };
}

function applyStepsToGrid(baseGrid, start, goal, steps, stepIndex, agentOnPath = false) {
  if (!baseGrid || baseGrid.length === 0 || !baseGrid[0]) {
    return [];
  }

  const g = copyGrid(baseGrid);
  const pathNodes = [];

  for (let i = 0; i <= stepIndex && i < steps.length; i++) {
    const step = steps[i];
    const [r, c] = step.node;

    if (r < 0 || r >= g.length || c < 0 || c >= g[0].length) continue;

    if (step.type === "open" && g[r][c] === CELL.EMPTY) g[r][c] = CELL.OPEN;
    if (step.type === "close" && g[r][c] !== CELL.START && g[r][c] !== CELL.GOAL) g[r][c] = CELL.CLOSED;
    if (step.type === "path") pathNodes.push([r, c]);
  }

  for (const [r, c] of pathNodes) {
    if (r < 0 || r >= g.length || c < 0 || c >= g[0].length) continue;

    if (!(r === start[0] && c === start[1]) && !(r === goal[0] && c === goal[1])) {
      g[r][c] = CELL.PATH;
    }
  }

  if (agentOnPath && pathNodes.length) {
    const [ar, ac] = pathNodes[pathNodes.length - 1];
    if (
      ar >= 0 && ar < g.length &&
      ac >= 0 && ac < g[0].length &&
      !(ar === goal[0] && ac === goal[1])
    ) {
      g[ar][ac] = CELL.AGENT;
    }
  }

  if (
    start[0] >= 0 && start[0] < g.length &&
    start[1] >= 0 && start[1] < g[0].length
  ) {
    g[start[0]][start[1]] = CELL.START;
  }

  if (
    goal[0] >= 0 && goal[0] < g.length &&
    goal[1] >= 0 && goal[1] < g[0].length
  ) {
    g[goal[0]][goal[1]] = CELL.GOAL;
  }

  return g;
}

function GridView({ grid, title, subtitle }) {
  const rows = grid.length;
  const cols = grid[0].length;
  const size = clamp(Math.floor(520 / Math.max(rows, cols)), 10, 24);

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">{title}</CardTitle>
        {subtitle ? <CardDescription>{subtitle}</CardDescription> : null}
      </CardHeader>
      <CardContent>
        <div
          className="grid gap-[2px] rounded-2xl bg-slate-300 p-2 w-fit max-w-full overflow-auto"
          style={{ gridTemplateColumns: `repeat(${cols}, ${size}px)` }}
        >
          {grid.flatMap((row, r) =>
            row.map((cell, c) => (
              <motion.div
                key={`${r}-${c}-${cell}`}
                layout
                initial={{ scale: 0.9, opacity: 0.9 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.12 }}
                className={`${colorMap[cell]} rounded-[4px] border border-white/30`}
                style={{ width: size, height: size }}
              />
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Legend() {
  const items = [
    [CELL.START, "Start"],
    [CELL.GOAL, "Goal"],
    [CELL.WALL, "Wall"],
    [CELL.OPEN, "Open set"],
    [CELL.CLOSED, "Visited"],
    [CELL.PATH, "Shortest path"],
    [CELL.AGENT, "Robot / particle"],
    [CELL.RANDOM, "Sampled state"],
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {items.map(([cell, label]) => (
        <div key={label} className="flex items-center gap-2 rounded-full border px-3 py-1 text-sm">
          <span className={`${colorMap[cell]} h-4 w-4 rounded-full border`} />
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}

function usePlannerGrid(rows, cols) {
  const [baseGrid, setBaseGrid] = useState([]);
  const [start, setStart] = useState([1, 1]);
  const [goal, setGoal] = useState([rows - 2, cols - 2]);

  useEffect(() => {
    const { grid, start, goal } = placeStartGoal(carveMaze(rows, cols));
    setBaseGrid(grid);
    setStart(start);
    setGoal(goal);
  }, [rows, cols]);

  function regenerateMaze() {
    const { grid, start, goal } = placeStartGoal(carveMaze(rows, cols));
    setBaseGrid(grid);
    setStart(start);
    setGoal(goal);
  }

  function regenerateRandom() {
    const { grid, start, goal } = placeStartGoal(createGrid(rows, cols, 0.26));
    setBaseGrid(grid);
    setStart(start);
    setGoal(goal);
  }

  return { baseGrid, setBaseGrid, start, goal, regenerateMaze, regenerateRandom };
}

function SearchDemo({ algorithm, label }) {
  const [rows, setRows] = useState(21);
  const [cols, setCols] = useState(21);
  const [speed, setSpeed] = useState(20);
  const { baseGrid, start, goal, regenerateMaze, regenerateRandom } = usePlannerGrid(rows, cols);
  const [viewGrid, setViewGrid] = useState([]);
  const [status, setStatus] = useState("Sẵn sàng");
  const [stats, setStats] = useState({ visited: 0, path: 0 });
  const runningRef = useRef(false);

  useEffect(() => {
    setViewGrid(applyStepsToGrid(baseGrid, start, goal, [], -1));
  }, [baseGrid, start, goal]);

  async function run() {
    if (!baseGrid.length) return;
    runningRef.current = true;
    setStatus("Đang tìm đường...");
    const result = genericSearch(baseGrid, start, goal, algorithm);
    setStats({ visited: result.visitedCount, path: result.path.length });

    for (let i = 0; i < result.steps.length && runningRef.current; i++) {
      setViewGrid(applyStepsToGrid(baseGrid, start, goal, result.steps, i, true));
      await sleep(speed);
    }

    if (result.found) {
      setStatus(`Đã tìm thấy đường đi bằng ${label}`);
    } else {
      setStatus("Không tìm được đường đi");
      setViewGrid(applyStepsToGrid(baseGrid, start, goal, result.steps, result.steps.length - 1));
    }
    runningRef.current = false;
  }

  function stop() {
    runningRef.current = false;
    setStatus("Đã dừng");
  }

  function resetView() {
    runningRef.current = false;
    setViewGrid(applyStepsToGrid(baseGrid, start, goal, [], -1));
    setStatus("Đã reset");
    setStats({ visited: 0, path: 0 });
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg"><Bot className="h-5 w-5" /> {label}</CardTitle>
            <CardDescription>Tạo mê cung ngẫu nhiên, thay đổi kích thước, và xem animation quá trình tìm đường.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div>
              <div className="mb-2 text-sm font-medium">Số hàng: {rows}</div>
              <Slider value={[rows]} min={11} max={35} step={2} onValueChange={(v) => setRows(v[0])} />
            </div>
            <div>
              <div className="mb-2 text-sm font-medium">Số cột: {cols}</div>
              <Slider value={[cols]} min={11} max={35} step={2} onValueChange={(v) => setCols(v[0])} />
            </div>
            <div>
              <div className="mb-2 text-sm font-medium">Tốc độ animation: {speed} ms</div>
              <Slider value={[speed]} min={5} max={120} step={5} onValueChange={(v) => setSpeed(v[0])} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button onClick={run} className="rounded-2xl"><Play className="mr-2 h-4 w-4" /> Chạy</Button>
              <Button variant="secondary" onClick={stop} className="rounded-2xl"><Pause className="mr-2 h-4 w-4" /> Dừng</Button>
              <Button variant="outline" onClick={resetView} className="rounded-2xl"><RotateCcw className="mr-2 h-4 w-4" /> Reset</Button>
              <Button variant="outline" onClick={regenerateMaze} className="rounded-2xl"><Shuffle className="mr-2 h-4 w-4" /> Mê cung</Button>
            </div>
            <Button variant="outline" onClick={regenerateRandom} className="w-full rounded-2xl">Tạo bản đồ chướng ngại vật ngẫu nhiên</Button>
            <div className="rounded-2xl border p-3 text-sm space-y-2">
              <div><strong>Trạng thái:</strong> {status}</div>
              <div><strong>Số ô đã duyệt:</strong> {stats.visited}</div>
              <div><strong>Độ dài đường đi:</strong> {stats.path}</div>
            </div>
            <Legend />
          </CardContent>
        </Card>
        {viewGrid.length ? <GridView grid={viewGrid} title={`Bản đồ ${label}`} subtitle="Xanh lá = điểm bắt đầu, đỏ = đích, vàng = đường đi tìm được" /> : null}
      </div>
    </div>
  );
}

function QLearningDemo() {
  const [rows, setRows] = useState(11);
  const [cols, setCols] = useState(11);
  const { baseGrid, start, goal, regenerateMaze } = usePlannerGrid(rows, cols);
  const [episodes, setEpisodes] = useState(250);
  const [speed, setSpeed] = useState(35);
  const [trainedPath, setTrainedPath] = useState([]);
  const [viewGrid, setViewGrid] = useState([]);
  const [summary, setSummary] = useState("Chưa huấn luyện");
  const qRef = useRef({});

  useEffect(() => {
    setTrainedPath([]);
    if (baseGrid.length) setViewGrid(applyStepsToGrid(baseGrid, start, goal, [], -1));
  }, [baseGrid, start, goal]);

  function stateKey(r, c) {
    return `${r},${c}`;
  }

  function nextState([r, c], action) {
    let nr = r;
    let nc = c;
    if (action === "UP") nr -= 1;
    if (action === "DOWN") nr += 1;
    if (action === "LEFT") nc -= 1;
    if (action === "RIGHT") nc += 1;
    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || baseGrid[nr][nc] === CELL.WALL) return [r, c];
    return [nr, nc];
  }

  function reward([r, c]) {
    if (r === goal[0] && c === goal[1]) return 100;
    return -1;
  }

  function getQ(s, a) {
    return qRef.current[`${s}|${a}`] ?? 0;
  }

  function setQ(s, a, val) {
    qRef.current[`${s}|${a}`] = val;
  }

  function bestAction(state) {
    let best = ACTIONS[0];
    let bestVal = -Infinity;
    for (const a of ACTIONS) {
      const v = getQ(state, a);
      if (v > bestVal) {
        bestVal = v;
        best = a;
      }
    }
    return best;
  }

  function train() {
    qRef.current = {};
    const alpha = 0.2;
    const gamma = 0.95;
    let epsilon = 0.8;

    for (let ep = 0; ep < episodes; ep++) {
      let current = [...start];
      for (let step = 0; step < rows * cols * 4; step++) {
        const s = stateKey(...current);
        const action = Math.random() < epsilon ? ACTIONS[randomInt(0, 3)] : bestAction(s);
        const ns = nextState(current, action);
        const nsKey = stateKey(...ns);
        const r = reward(ns);
        const maxNext = Math.max(...ACTIONS.map((a) => getQ(nsKey, a)));
        const updated = getQ(s, action) + alpha * (r + gamma * maxNext - getQ(s, action));
        setQ(s, action, updated);
        current = ns;
        if (ns[0] === goal[0] && ns[1] === goal[1]) break;
      }
      epsilon *= 0.992;
    }

    const path = [start];
    let current = [...start];
    const seen = new Set([keyOf(...current)]);
    for (let i = 0; i < rows * cols * 3; i++) {
      if (current[0] === goal[0] && current[1] === goal[1]) break;
      const act = bestAction(stateKey(...current));
      const nxt = nextState(current, act);
      const k = keyOf(...nxt);
      path.push(nxt);
      current = nxt;
      if (seen.has(k)) break;
      seen.add(k);
    }

    setTrainedPath(path);
    setSummary(`Huấn luyện xong với ${episodes} episodes. Độ dài policy path: ${path.length}`);
  }

  async function animatePolicy() {
    if (!trainedPath.length) return;
    for (let i = 0; i < trainedPath.length; i++) {
      const g = copyGrid(baseGrid);
      for (let p = 0; p <= i; p++) {
        const [r, c] = trainedPath[p];
        if (!(r === start[0] && c === start[1]) && !(r === goal[0] && c === goal[1])) g[r][c] = CELL.PATH;
      }
      const [ar, ac] = trainedPath[i];
      if (!(ar === goal[0] && ac === goal[1])) g[ar][ac] = CELL.AGENT;
      g[start[0]][start[1]] = CELL.START;
      g[goal[0]][goal[1]] = CELL.GOAL;
      setViewGrid(g);
      await sleep(speed);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg"><Brain className="h-5 w-5" /> GridWorld với Q-learning</CardTitle>
          <CardDescription>Huấn luyện tác nhân học tăng cường để tìm đường tới đích sau khi khám phá môi trường.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div>
            <div className="mb-2 text-sm font-medium">Số hàng: {rows}</div>
            <Slider value={[rows]} min={9} max={19} step={2} onValueChange={(v) => setRows(v[0])} />
          </div>
          <div>
            <div className="mb-2 text-sm font-medium">Số cột: {cols}</div>
            <Slider value={[cols]} min={9} max={19} step={2} onValueChange={(v) => setCols(v[0])} />
          </div>
          <div>
            <div className="mb-2 text-sm font-medium">Số episodes: {episodes}</div>
            <Slider value={[episodes]} min={50} max={1200} step={50} onValueChange={(v) => setEpisodes(v[0])} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Button onClick={train} className="rounded-2xl">Huấn luyện</Button>
            <Button variant="secondary" onClick={animatePolicy} className="rounded-2xl">Chạy policy</Button>
            <Button variant="outline" onClick={regenerateMaze} className="col-span-2 rounded-2xl">Tạo mê cung mới</Button>
          </div>
          <div className="rounded-2xl border p-3 text-sm">{summary}</div>
          <Legend />
        </CardContent>
      </Card>
      {viewGrid.length ? <GridView grid={viewGrid} title="Q-learning policy" subtitle="Đường màu vàng là đường mà agent học được" /> : null}
    </div>
  );
}

function LogicDemo() {
  const [isDark, setIsDark] = useState(true);
  const [someoneHome, setSomeoneHome] = useState(true);
  const [motion, setMotion] = useState(false);
  const [hot, setHot] = useState(false);
  const [doorOpen, setDoorOpen] = useState(false);

  const facts = useMemo(() => {
    const f = [];
    if (isDark) f.push("Trời tối");
    if (someoneHome) f.push("Có người ở nhà");
    if (motion) f.push("Có chuyển động");
    if (hot) f.push("Nhiệt độ cao");
    if (doorOpen) f.push("Cửa đang mở");
    return f;
  }, [isDark, someoneHome, motion, hot, doorOpen]);

  const actions = useMemo(() => {
    const out = [];
    if (isDark && someoneHome) out.push("Bật đèn phòng khách");
    if (!someoneHome) out.push("Bật chế độ tiết kiệm điện");
    if (motion && isDark) out.push("Bật đèn hành lang");
    if (hot && someoneHome) out.push("Bật điều hòa");
    if (doorOpen && !someoneHome) out.push("Gửi cảnh báo an ninh");
    if (!out.length) out.push("Không cần hành động nào");
    return out;
  }, [isDark, someoneHome, motion, hot, doorOpen]);

  const rules = [
    "Nếu trời tối và có người ở nhà => bật đèn phòng khách",
    "Nếu có chuyển động và trời tối => bật đèn hành lang",
    "Nếu nhiệt độ cao và có người ở nhà => bật điều hòa",
    "Nếu không có người ở nhà => bật chế độ tiết kiệm điện",
    "Nếu cửa mở và không có người ở nhà => gửi cảnh báo",
  ];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg"><Home className="h-5 w-5" /> Tự động hóa nhà thông minh</CardTitle>
          <CardDescription>Suy luận logic dạng luật if-then.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            ["Trời tối", isDark, setIsDark],
            ["Có người ở nhà", someoneHome, setSomeoneHome],
            ["Có chuyển động", motion, setMotion],
            ["Nhiệt độ cao", hot, setHot],
            ["Cửa đang mở", doorOpen, setDoorOpen],
          ].map(([label, value, setter]) => (
            <div key={label} className="flex items-center justify-between rounded-2xl border p-3">
              <span>{label}</span>
              <Switch checked={value} onCheckedChange={setter} />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg">Kết quả suy luận</CardTitle>
          <CardDescription>Facts hiện tại và các hành động được kích hoạt.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="mb-2 text-sm font-medium">Facts</div>
            <div className="flex flex-wrap gap-2">
              {facts.map((f) => <Badge key={f} variant="secondary" className="rounded-full">{f}</Badge>)}
            </div>
          </div>
          <div>
            <div className="mb-2 text-sm font-medium">Actions</div>
            <div className="space-y-2">
              {actions.map((a) => (
                <div key={a} className="rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm dark:border-emerald-900 dark:bg-emerald-950/40">
                  {a}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="mb-2 text-sm font-medium">Luật suy luận</div>
            <div className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
              {rules.map((r) => <div key={r} className="rounded-xl border p-2">{r}</div>)}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function StripsDemo() {
  const [goal, setGoal] = useState("deliver");

  const domains = {
    tea: {
      initial: ["have_water", "have_kettle", "have_cup", "have_tea_bag"],
      goal: ["tea_ready"],
      plan: [
        { name: "boil_water", pre: ["have_water", "have_kettle"], add: ["water_boiled"] },
        { name: "put_tea_bag_in_cup", pre: ["have_cup", "have_tea_bag"], add: ["tea_bag_in_cup"] },
        { name: "pour_boiled_water", pre: ["water_boiled", "tea_bag_in_cup"], add: ["tea_ready"] },
      ],
    },
    deliver: {
      initial: ["robot_at_A", "package_at_A", "path_A_B", "gripper_empty"],
      goal: ["package_at_B"],
      plan: [
        { name: "pickup_package", pre: ["robot_at_A", "package_at_A", "gripper_empty"], add: ["holding_package"], del: ["gripper_empty", "package_at_A"] },
        { name: "move_A_to_B", pre: ["robot_at_A", "path_A_B"], add: ["robot_at_B"], del: ["robot_at_A"] },
        { name: "drop_package", pre: ["robot_at_B", "holding_package"], add: ["package_at_B", "gripper_empty"], del: ["holding_package"] },
      ],
    },
  };

  const selected = domains[goal];

  const execution = useMemo(() => {
    let facts = new Set(selected.initial);
    const logs = [];
    for (const action of selected.plan) {
      const ok = action.pre.every((p) => facts.has(p));
      logs.push({ action: action.name, ok, before: [...facts] });
      if (ok) {
        (action.del || []).forEach((d) => facts.delete(d));
        (action.add || []).forEach((a) => facts.add(a));
      }
      logs[logs.length - 1].after = [...facts];
    }
    const success = selected.goal.every((g) => facts.has(g));
    return { logs, finalFacts: [...facts], success };
  }, [goal]);

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg"><Target className="h-5 w-5" /> STRIPS / PDDL mini planner</CardTitle>
          <CardDescription>Biểu diễn precondition và effect, sau đó suy ra chuỗi hành động.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="mb-2 text-sm font-medium">Chọn bài toán</div>
            <Select value={goal} onValueChange={setGoal}>
              <SelectTrigger className="rounded-2xl"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="deliver">Robot giao kiện hàng A → B</SelectItem>
                <SelectItem value="tea">Pha trà</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="rounded-2xl border p-3 text-sm">
            <div><strong>Initial state:</strong> {selected.initial.join(", ")}</div>
            <div className="mt-2"><strong>Goal:</strong> {selected.goal.join(", ")}</div>
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg">Kế hoạch suy luận</CardTitle>
          <CardDescription>{execution.success ? "Đã đạt mục tiêu" : "Chưa đạt mục tiêu"}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {execution.logs.map((log, idx) => (
            <div key={log.action} className="rounded-2xl border p-3">
              <div className="font-medium">Bước {idx + 1}: {log.action}</div>
              <div className="mt-2">Trước: {log.before.join(", ")}</div>
              <div>Thực hiện được: {log.ok ? "Có" : "Không"}</div>
              <div>Sau: {log.after.join(", ")}</div>
            </div>
          ))}
          <div className="rounded-2xl border border-fuchsia-200 bg-fuchsia-50 p-3 dark:border-fuchsia-900 dark:bg-fuchsia-950/40">
            <strong>Final facts:</strong> {execution.finalFacts.join(", ")}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function LocalizationDemo() {
  const [rows, setRows] = useState(15);
  const [cols, setCols] = useState(15);
  const { baseGrid, start, goal, regenerateRandom } = usePlannerGrid(rows, cols);
  const [robot, setRobot] = useState([1, 1]);
  const [particles, setParticles] = useState([]);
  const [viewGrid, setViewGrid] = useState([]);
  const [stepsCount, setStepsCount] = useState(0);

  useEffect(() => {
    if (!baseGrid.length) return;
    const free = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) if (baseGrid[r][c] !== CELL.WALL) free.push([r, c]);
    }
    setRobot(start);
    setParticles(Array.from({ length: 120 }, () => free[randomInt(0, free.length - 1)]));
    setViewGrid(applyParticleView(baseGrid, start, goal, start, []));
    setStepsCount(0);
  }, [baseGrid, start, goal, rows, cols]);

  function sense(pos) {
    const [r, c] = pos;
    return DIRS4.map(([dr, dc]) => {
      const nr = r + dr;
      const nc = c + dc;
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) return 1;
      return baseGrid[nr][nc] === CELL.WALL ? 1 : 0;
    });
  }

  function scoreParticle(p, measurement) {
    const s = sense(p);
    let score = 1;
    for (let i = 0; i < 4; i++) {
      score *= s[i] === measurement[i] ? 0.9 : 0.1;
    }
    return score;
  }

  function movePos(pos, action) {
    let [r, c] = pos;
    if (action === "UP") r -= 1;
    if (action === "DOWN") r += 1;
    if (action === "LEFT") c -= 1;
    if (action === "RIGHT") c += 1;
    if (r < 0 || r >= rows || c < 0 || c >= cols || baseGrid[r][c] === CELL.WALL) return pos;
    return [r, c];
  }

  function resample(weighted) {
    const total = weighted.reduce((s, x) => s + x.w, 0) || 1;
    const normalized = weighted.map((x) => ({ ...x, w: x.w / total }));
    const newParticles = [];
    for (let i = 0; i < weighted.length; i++) {
      let pick = Math.random();
      let acc = 0;
      for (const item of normalized) {
        acc += item.w;
        if (pick <= acc) {
          newParticles.push(item.p);
          break;
        }
      }
    }
    return newParticles;
  }

  function applyParticleView(grid, start, goal, robotPos, particles) {
    const g = copyGrid(grid);
    const count = {};
    for (const p of particles) {
      const k = keyOf(...p);
      count[k] = (count[k] || 0) + 1;
    }
    for (const k of Object.keys(count)) {
      const [r, c] = parseKey(k);
      if (g[r][c] === CELL.EMPTY) g[r][c] = CELL.RANDOM;
    }
    const [rr, rc] = robotPos;
    g[rr][rc] = CELL.AGENT;
    g[start[0]][start[1]] = CELL.START;
    g[goal[0]][goal[1]] = CELL.GOAL;
    return g;
  }

  function stepOnce() {
    const action = ACTIONS[randomInt(0, 3)];
    const movedRobot = movePos(robot, action);
    const measurement = sense(movedRobot);
    const movedParticles = particles.map((p) => movePos(p, Math.random() < 0.8 ? action : ACTIONS[randomInt(0, 3)]));
    const weighted = movedParticles.map((p) => ({ p, w: scoreParticle(p, measurement) }));
    const newParticles = resample(weighted);
    setRobot(movedRobot);
    setParticles(newParticles);
    setViewGrid(applyParticleView(baseGrid, start, goal, movedRobot, newParticles));
    setStepsCount((x) => x + 1);
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg"><Map className="h-5 w-5" /> Monte Carlo Localization</CardTitle>
          <CardDescription>Các hạt màu cam biểu diễn các giả thuyết về vị trí robot.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div>
            <div className="mb-2 text-sm font-medium">Số hàng: {rows}</div>
            <Slider value={[rows]} min={11} max={25} step={2} onValueChange={(v) => setRows(v[0])} />
          </div>
          <div>
            <div className="mb-2 text-sm font-medium">Số cột: {cols}</div>
            <Slider value={[cols]} min={11} max={25} step={2} onValueChange={(v) => setCols(v[0])} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Button onClick={stepOnce} className="rounded-2xl">Bước tiếp</Button>
            <Button variant="outline" onClick={regenerateRandom} className="rounded-2xl">Bản đồ mới</Button>
          </div>
          <div className="rounded-2xl border p-3 text-sm">
            <div><strong>Số bước:</strong> {stepsCount}</div>
            <div><strong>Ý nghĩa:</strong> qua mỗi bước, particle filter cập nhật và hội tụ dần quanh robot thật.</div>
          </div>
          <Legend />
        </CardContent>
      </Card>
      {viewGrid.length ? <GridView grid={viewGrid} title="Particle filter view" subtitle="Tím = robot thật, cam = particle hypotheses" /> : null}
    </div>
  );
}

export default function AIRobotPlanningLab() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-fuchsia-50 p-4 text-slate-900 dark:from-slate-950 dark:via-slate-950 dark:to-slate-900 dark:text-slate-100 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid gap-4 rounded-[28px] border bg-white/80 p-6 shadow-sm backdrop-blur dark:bg-slate-900/80 md:grid-cols-[1.4fr_0.8fr]"
        >
          <div>
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm">
              <Sparkles className="h-4 w-4" /> AI Planning + Robotics Lab
            </div>
            <h1 className="text-3xl font-bold tracking-tight md:text-4xl">Bộ mô phỏng đầy đủ cho 7 bài toán AI và robot</h1>
            <p className="mt-3 max-w-3xl text-base text-slate-600 dark:text-slate-300">
              Gồm A*, Dijkstra, Greedy search, Random search, GridWorld với Q-learning, suy luận logic cho smart home,
              STRIPS/PDDL mini planner và Monte Carlo localization. Có animation, mê cung ngẫu nhiên và thay đổi kích thước bản đồ.
            </p>
          </div>
          <Card className="rounded-3xl border-dashed shadow-none">
            <CardHeader>
              <CardTitle className="text-lg">Các module có sẵn</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {["A*", "Dijkstra", "Greedy", "Random Search", "Q-learning", "Logic", "STRIPS/PDDL", "MCL"].map((x) => (
                <Badge key={x} className="rounded-full px-3 py-1">{x}</Badge>
              ))}
            </CardContent>
          </Card>
        </motion.div>

        <Tabs defaultValue="astar" className="space-y-4">
          <TabsList className="flex h-auto flex-wrap justify-start gap-2 rounded-2xl bg-transparent p-0">
            <TabsTrigger value="astar" className="rounded-2xl border px-4 py-2">1. A*</TabsTrigger>
            <TabsTrigger value="greedy" className="rounded-2xl border px-4 py-2">2. Greedy</TabsTrigger>
            <TabsTrigger value="random" className="rounded-2xl border px-4 py-2">3. Random</TabsTrigger>
            <TabsTrigger value="gridworld" className="rounded-2xl border px-4 py-2">4. GridWorld + Q-learning</TabsTrigger>
            <TabsTrigger value="logic" className="rounded-2xl border px-4 py-2">5. Smart Home Logic</TabsTrigger>
            <TabsTrigger value="strips" className="rounded-2xl border px-4 py-2">6. STRIPS / PDDL</TabsTrigger>
            <TabsTrigger value="mcl" className="rounded-2xl border px-4 py-2">7. Localization</TabsTrigger>
            <TabsTrigger value="dijkstra" className="rounded-2xl border px-4 py-2">Bonus: Dijkstra</TabsTrigger>
          </TabsList>

          <TabsContent value="astar"><SearchDemo algorithm="astar" label="A* Search" /></TabsContent>
          <TabsContent value="greedy"><SearchDemo algorithm="greedy" label="Greedy Best-First Search" /></TabsContent>
          <TabsContent value="random"><SearchDemo algorithm="random" label="Random Search" /></TabsContent>
          <TabsContent value="gridworld"><QLearningDemo /></TabsContent>
          <TabsContent value="logic"><LogicDemo /></TabsContent>
          <TabsContent value="strips"><StripsDemo /></TabsContent>
          <TabsContent value="mcl"><LocalizationDemo /></TabsContent>
          <TabsContent value="dijkstra"><SearchDemo algorithm="dijkstra" label="Dijkstra Search" /></TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
