// Interactive Agent Replay UI (placeholder)
import React from "react";

type ReplayEntry = {
  timestamp: string;
  signal: string;
  reasoning: string;
  strength: number;
};

const ReplayAgent: React.FC<{ entries: ReplayEntry[] }> = ({ entries }) => {
  return (
    <div>
      <h2>Agent Replay</h2>
      {entries.map((e, idx) => (
        <div key={idx} style={{ border: "1px solid #444", marginBottom: 8, padding: 8 }}>
          <div>{e.timestamp}</div>
          <div>Signal: {e.signal}</div>
          <div>Reasoning: {e.reasoning}</div>
          <div>Strength: {e.strength}</div>
        </div>
      ))}
    </div>
  );
};

export default ReplayAgent;
