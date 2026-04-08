export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx,mdx}", "./.storybook/**/*.{js,jsx,ts,tsx,mdx}"],
  theme: {
    extend: {
      screens: {
        tablet: "1024px"
      },
      colors: {
        surface: "hsl(var(--surface) / <alpha-value>)",
        elevated: "hsl(var(--elevated) / <alpha-value>)",
        line: "hsl(var(--line) / <alpha-value>)",
        text: "hsl(var(--text) / <alpha-value>)",
        muted: "hsl(var(--muted) / <alpha-value>)",
        primary: "hsl(var(--primary) / <alpha-value>)",
        accent: "hsl(var(--accent) / <alpha-value>)",
        success: "hsl(var(--success) / <alpha-value>)",
        warning: "hsl(var(--warning) / <alpha-value>)",
        danger: "hsl(var(--danger) / <alpha-value>)",
        background: "var(--background)",
        foreground: "var(--foreground)",
        card: "var(--card)",
        border: "var(--border)",
        secondary: "var(--secondary)",
        "muted-foreground": "var(--muted-foreground)",
        "primary-foreground": "var(--primary-foreground)"
      },
      boxShadow: {
        panel: "0 8px 24px rgba(2, 12, 24, 0.22)",
        glow: "0 0 0 1px rgba(125, 211, 252, 0.15), 0 0 32px rgba(20, 184, 166, 0.18)"
      },
      borderRadius: {
        panel: "var(--radius-panel)"
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "'Source Sans 3'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"]
      }
    }
  },
  plugins: []
};
