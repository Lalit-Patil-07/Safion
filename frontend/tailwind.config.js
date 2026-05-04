/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#F54F00',      // Signal Orange
        background: '#09090E',   // Void
        card: '#181826',         // Node Inactive / Card
        'card-secondary': '#131322', // Ink 2
        border: '#1C1C2E',       // Boundary
        text: {
          DEFAULT: '#EDEDF4',    // Frost
          secondary: '#7878A0',  // Text Mid
          tertiary: '#363650',   // Text Lo
        },
        accent: {
          green: '#48BB78',
          yellow: '#F54F00',   // Frigate Orange -> mapped to Signal Orange just in case
          red: '#EF4444',
        },
      },
      fontFamily: {
        sans: ['Space Grotesk', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
