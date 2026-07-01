import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("App", () => {
  it("renders the workbench as the first screen", () => {
    render(<App />);
    expect(screen.getByLabelText("Atlas tools")).toBeInTheDocument();
    expect(screen.getByText("Analyze")).toBeInTheDocument();
    expect(screen.getByText("Guide")).toBeInTheDocument();
    expect(screen.getByDisplayValue("http://127.0.0.1:1234/v1")).toBeInTheDocument();
    expect(screen.getByText("Refresh models")).toBeInTheDocument();
    expect(screen.getByText("3D Lineup")).toBeInTheDocument();
    expect(screen.getByText("Load a still image to begin camera lineup.")).toBeInTheDocument();
  });
});
