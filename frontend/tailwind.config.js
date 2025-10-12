/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#5a51a2',      // Frigate Purple
        background: '#1c1e26',   // Frigate Dark Background
        card: '#2d3748',         // Frigate Card Background
        'card-secondary': '#1f2937', // Slightly darker card variant
        border: '#4A5568',       // Borders and dividers
        text: {
          DEFAULT: '#E2E8F0',    // Default text color
          secondary: '#A0AEC0',  // Lighter, secondary text
        },
        accent: {
          green: '#48BB78',
          yellow: '#f09239',   // Frigate Orange
          red: '#bd3a06',      // Frigate Red
        },
      },
    },
  },
  plugins: [],
}