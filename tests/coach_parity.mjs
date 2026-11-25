// Runs phrases through the browser keyword parser (docs/js/coach.js) and prints
// the tactics as JSON. Used by tests/test_web_coach.py.
import { readFileSync } from "node:fs";
import { clamp, keywordTactic, setCoachData } from "../docs/js/coach.js";

const data = JSON.parse(readFileSync(new URL("../docs/js/coach.json", import.meta.url), "utf8"));
setCoachData(data);
const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
console.log(JSON.stringify({
  keywords: input.phrases.map((t) => keywordTactic(t)),
  clamp: input.raw.map((z) => clamp(z)),
}));
