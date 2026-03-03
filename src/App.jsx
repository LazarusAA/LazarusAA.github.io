import React, { useLayoutEffect, useRef, useState, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { ArrowRight, Terminal, Cpu, Network, Globe, MousePointer2 } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

// ==========================================
// 1. NAVBAR - "The Floating Island"
// ==========================================
function Navbar() {
  const navRef = useRef(null);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="fixed top-6 left-0 w-full z-50 flex justify-center px-4">
      <nav
        ref={navRef}
        className={`transition-all duration-500 ease-[cubic-bezier(0.25,0.46,0.45,0.94)] flex items-center justify-between px-6 py-3 rounded-full border text-dark ${isScrolled
          ? 'bg-background/80 backdrop-blur-xl border-dark/10 shadow-lg'
          : 'bg-transparent border-transparent'
          }`}
        style={{ width: 'min(90%, 800px)' }}
      >
        <div className="font-sans font-bold text-lg tracking-tight uppercase">
          Lazarus
        </div>

        <div className="hidden md:flex items-center gap-8 font-mono text-sm">
          <a href="#hardware" className="hover:-translate-y-[1px] transition-transform">Hardware</a>
          <a href="#digital" className="hover:-translate-y-[1px] transition-transform">Digital</a>
          <a href="#bridge" className="hover:-translate-y-[1px] transition-transform">Bridge</a>
        </div>

        <button className="magnetic-btn bg-accent text-primary px-5 py-2 rounded-full font-sans font-semibold text-sm h-10 overflow-hidden group">
          <span className="magnetic-btn-text flex items-center gap-2 group-hover:text-primary transition-colors">
            Access <ArrowRight size={16} />
          </span>
        </button>
      </nav>
    </div>
  );
}

// ==========================================
// 2. HERO SECTION - "The Opening Shot"
// ==========================================
function Hero() {
  const containerRef = useRef(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.hero-text', {
        y: 40,
        opacity: 0,
        duration: 1,
        stagger: 0.08,
        ease: 'power3.out',
        delay: 0.2
      });
      gsap.from('.hero-btn', {
        y: 40,
        opacity: 0,
        duration: 1,
        ease: 'power3.out',
        delay: 0.6
      });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={containerRef}
      className="relative h-[100dvh] w-full flex items-end pb-24 px-6 md:px-16 overflow-hidden bg-dark text-primary"
    >
      <div
        className="absolute inset-0 z-0 bg-cover bg-center bg-no-repeat grayscale-[20%]"
        style={{
          backgroundImage: 'url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop")',
        }}
      />
      <div className="absolute inset-0 z-10 bg-gradient-to-t from-dark/95 via-dark/60 to-transparent" />

      <div className="relative z-20 w-full max-w-6xl mx-auto">
        <div className="font-mono text-sm tracking-widest text-primary/70 mb-6 hero-text uppercase flex items-center gap-3">
          <Terminal size={14} /> System Initialized
        </div>

        <h1 className="flex flex-col gap-0 leading-[0.9]">
          <span className="hero-text font-sans font-bold text-5xl md:text-7xl tracking-tighter uppercase text-primary">
            ACCESS the
          </span>
          <span className="hero-text font-drama italic text-7xl md:text-[140px] text-accent tracking-tight pr-4">
            Blueprint.
          </span>
        </h1>

        <p className="hero-text mt-8 font-sans text-xl md:text-2xl max-w-2xl text-primary/80 leading-snug">
          Scaling physical and digital autonomy for the next era of Latin America.
        </p>

        <div className="mt-12 hero-btn">
          <button className="magnetic-btn bg-accent text-primary px-8 py-4 rounded-full font-sans font-bold text-lg overflow-hidden group border border-transparent hover:border-accent">
            <span className="magnetic-btn-text flex items-center gap-3">
              Access the Blueprint <ArrowRight size={20} />
            </span>
          </button>
        </div>
      </div>
    </section>
  );
}

// ==========================================
// 3. FEATURES SECTIONS - "Interactive Functional Artifacts"
// ==========================================

// Card 1: Diagnostic Shuffler
function HardwareShuffler() {
  const [cards, setCards] = useState([
    { id: 1, title: 'Eastern Mfg.', val: 'OPTIMIZED' },
    { id: 2, title: 'Robotics', val: 'SCALED' },
    { id: 3, title: 'Megaprojects', val: 'DECODED' }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCards(prev => {
        const newArr = [...prev];
        const last = newArr.pop();
        newArr.unshift(last);
        return newArr;
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="feature-card min-h-[400px] flex flex-col pt-8">
      <div className="flex items-center gap-3 mb-6">
        <Cpu className="text-accent" size={24} />
        <h3 className="font-sans font-bold text-2xl uppercase tracking-tighter">The Hardware Frontier</h3>
      </div>
      <p className="font-mono text-sm text-dark/70 mb-8 max-w-sm">
        Decoding Eastern manufacturing, robotics, and the megaprojects shaping the physical world.
      </p>

      <div className="relative flex-1 mt-4">
        {cards.map((c, i) => {
          const isTop = i === 0;
          return (
            <div
              key={c.id}
              className="absolute w-[90%] left-[5%] p-4 rounded-xl border border-dark/10 bg-background shadow-md flex justify-between items-center transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]"
              style={{
                top: `${i * 20}px`,
                scale: 1 - (i * 0.05),
                opacity: 1 - (i * 0.2),
                zIndex: 10 - i,
              }}
            >
              <span className="font-sans font-bold text-lg">{c.title}</span>
              <span className={`font-mono text-xs ${isTop ? 'text-accent' : 'text-dark/50'}`}>
                [{c.val}]
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Card 2: Telemetry Typewriter
function DigitalTypewriter() {
  const messages = [
    "Architecting AI agents...",
    "Scaling output x100...",
    "Deploying autonomous communities..."
  ];
  const [text, setText] = useState('');
  const [msgIdx, setMsgIdx] = useState(0);

  useEffect(() => {
    let currentText = '';
    let charIdx = 0;
    const targetText = messages[msgIdx];

    const typeInterval = setInterval(() => {
      currentText += targetText[charIdx];
      setText(currentText);
      charIdx++;
      if (charIdx >= targetText.length) {
        clearInterval(typeInterval);
        setTimeout(() => {
          setMsgIdx((prev) => (prev + 1) % messages.length);
        }, 2000);
      }
    }, 50);

    return () => clearInterval(typeInterval);
  }, [msgIdx]);

  return (
    <div className="feature-card min-h-[400px] flex flex-col pt-8">
      <div className="flex items-center gap-3 mb-6">
        <Network className="text-accent" size={24} />
        <h3 className="font-sans font-bold text-2xl uppercase tracking-tighter">Digital Leverage</h3>
      </div>
      <div className="flex items-center gap-2 mb-8">
        <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
        <span className="font-mono text-xs text-accent uppercase">Live Telemetry</span>
      </div>

      <p className="font-mono text-sm text-dark/70 mb-auto pr-4 leading-relaxed">
        Building the architecture for AI agents and autonomous communities that scale your time.
      </p>

      <div className="mt-8 p-4 rounded-xl bg-dark font-mono text-sm text-primary h-24">
        <span className="text-accent mr-2">{'>'}</span>
        {text}
        <span className="animate-pulse ml-1 inline-block w-2 bg-accent h-4 align-middle" />
      </div>
    </div>
  );
}

// Card 3: Cursor Protocol Scheduler
function BridgeScheduler() {
  const days = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
  const gridRef = useRef(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ repeat: -1, repeatDelay: 1 });
      tl.to('.anim-cursor', { x: 50, y: 30, duration: 1, ease: 'power2.inOut' })
        .to('.anim-cursor', { scale: 0.8, duration: 0.1, ease: 'power1.in' }) // Click
        .to('.day-cell-3', { backgroundColor: '#E63B2E', color: '#F5F3EE', duration: 0.1 })
        .to('.anim-cursor', { scale: 1, duration: 0.1, ease: 'power1.out' }) // Release
        .to('.anim-cursor', { x: 180, y: 80, duration: 1, ease: 'power2.inOut', delay: 0.5 })
        .to('.anim-cursor', { scale: 0.8, duration: 0.1, ease: 'power1.in' }) // Click save
        .to('.save-btn', { backgroundColor: '#E63B2E', color: '#F5F3EE', duration: 0.1 })
        .to('.anim-cursor', { scale: 1, duration: 0.1, ease: 'power1.out' })
        .to('.anim-cursor', { opacity: 0, duration: 0.5, delay: 0.5 })
        .set('.day-cell-3', { backgroundColor: 'transparent', color: 'inherit' })
        .set('.save-btn', { backgroundColor: 'transparent', color: 'inherit' })
        .set('.anim-cursor', { x: 0, y: 0, opacity: 1 });
    }, gridRef);

    return () => ctx.revert();
  }, []);

  return (
    <div className="feature-card min-h-[400px] flex flex-col pt-8">
      <div className="flex items-center gap-3 mb-6">
        <Globe className="text-accent" size={24} />
        <h3 className="font-sans font-bold text-2xl uppercase tracking-tighter">The Strategic Bridge</h3>
      </div>
      <p className="font-mono text-sm text-dark/70 mb-auto max-w-sm">
        Connecting Latin American ambition with Chinese industrial efficiency to build the new economy.
      </p>

      <div ref={gridRef} className="relative mt-8 p-6 rounded-xl border border-dark/10 bg-background">
        <div className="flex justify-between mb-6">
          {days.map((d, i) => (
            <div key={i} className={`w-8 h-8 flex items-center justify-center rounded-md font-mono text-xs border border-dark/5 day-cell-${i}`}>
              {d}
            </div>
          ))}
        </div>
        <div className="flex justify-end">
          <div className="save-btn px-4 py-1 border border-dark/20 rounded-full font-mono text-xs transition-colors">
            SYNC PROTOCOL
          </div>
        </div>

        {/* Animated Cursor */}
        <MousePointer2
          className="anim-cursor absolute top-0 left-0 text-dark"
          size={24}
          fill="#111111"
          style={{ transform: 'translate(0,0)' }}
        />
      </div>
    </div>
  );
}

function Features() {
  const containerRef = useRef(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.stagger-card', {
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top 70%',
        },
        y: 60,
        opacity: 0,
        duration: 1,
        stagger: 0.15,
        ease: 'power3.out'
      });
    }, containerRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={containerRef} className="py-24 px-6 md:px-16 w-full max-w-7xl mx-auto" id="hardware">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="stagger-card">
          <HardwareShuffler />
        </div>
        <div className="stagger-card">
          <DigitalTypewriter />
        </div>
        <div className="stagger-card">
          <BridgeScheduler />
        </div>
      </div>
    </section>
  );
}

// ==========================================
// 4. PHILOSOPHY - "The Manifesto"
// ==========================================
function Philosophy() {
  const containerRef = useRef(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      // Parallax Background
      gsap.to('.parallax-bg', {
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
        },
        y: '20%',
        ease: 'none'
      });

      // Text Reveal
      gsap.from('.manifesto-text', {
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top 60%',
        },
        y: 50,
        opacity: 0,
        duration: 1.2,
        stagger: 0.2,
        ease: 'power3.out'
      });
    }, containerRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={containerRef} className="relative min-h-[80vh] w-full flex items-center py-24 px-6 md:px-16 overflow-hidden bg-dark text-primary">
      {/* Background Parallax Texture */}
      <div
        className="parallax-bg absolute -top-[10%] -bottom-[10%] left-0 right-0 z-0 bg-cover bg-center opacity-20 grayscale"
        style={{
          backgroundImage: 'url("https://images.unsplash.com/photo-1497366216548-37526070297c?q=80&w=2069&auto=format&fit=crop")',
        }}
      />
      <div className="absolute inset-0 z-10 bg-dark/70" />

      <div className="relative z-20 w-full max-w-5xl mx-auto flex flex-col gap-12">
        <div className="manifesto-text">
          <p className="font-sans text-xl md:text-3xl text-primary/60 font-medium">
            Most infrastructure building focuses on: <span className="text-primary italic">isolated, theoretical models.</span>
          </p>
        </div>
        <div className="manifesto-text">
          <h2 className="flex flex-col gap-0 leading-[0.9]">
            <span className="font-sans font-bold text-3xl md:text-5xl uppercase text-primary">
              We focus on:
            </span>
            <span className="font-drama italic text-5xl md:text-[100px] text-accent tracking-tighter mt-2">
              Integrated, Autonomous Execution.
            </span>
          </h2>
        </div>
      </div>
    </section>
  );
}

// ==========================================
// 5. PROTOCOL - "Sticky Stacking Archive"
// ==========================================
function Protocol() {
  const containerRef = useRef(null);

  const steps = [
    {
      num: '01',
      title: 'Structural Synthesis',
      desc: 'Merging hardware supply chains with cutting-edge protocol logic.',
      Anim: () => (
        <div className="absolute inset-0 flex items-center justify-center opacity-20">
          <svg viewBox="0 0 100 100" className="w-[120%] h-[120%] animate-[spin_20s_linear_infinite]">
            <path d="M50 5 L95 25 L95 75 L50 95 L5 75 L5 25 Z" fill="none" stroke="currentColor" strokeWidth="0.5" />
            <path d="M50 15 L85 30 L85 70 L50 85 L15 70 L15 30 Z" fill="none" stroke="currentColor" strokeWidth="0.5" />
            <circle cx="50" cy="50" r="10" fill="none" stroke="currentColor" strokeWidth="0.5" />
          </svg>
        </div>
      )
    },
    {
      num: '02',
      title: 'Agentic Deployment',
      desc: 'Deploying autonomous agents to optimize and scale operational output.',
      Anim: () => (
        <div className="absolute inset-0 flex flex-col justify-between opacity-20 py-10">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="w-full h-px bg-currentColor relative overflow-hidden">
              <div
                className="absolute top-0 left-0 w-1/3 h-full bg-accent blur-sm"
                style={{
                  animation: `scan 3s linear infinite`,
                  animationDelay: `${i * 0.5}s`
                }}
              />
            </div>
          ))}
          <style dangerouslySetInnerHTML={{
            __html: `
            @keyframes scan { 0% { transform: translateX(-100%); } 100% { transform: translateX(300%); } }
          `}} />
        </div>
      )
    },
    {
      num: '03',
      title: 'Continuous Scaling',
      desc: 'Feedback loops driving exponential growth across physical borders.',
      Anim: () => (
        <div className="absolute inset-0 flex items-center justify-center opacity-20 overflow-hidden">
          <svg viewBox="0 0 100 20" className="w-full h-full stroke-currentColor" strokeWidth="0.2" fill="none">
            <path
              d="M0 10 L30 10 L35 0 L40 20 L45 0 L50 20 L55 5 L60 10 L100 10"
              className="dash-anim"
              strokeDasharray="100"
              strokeDashoffset="100"
            />
          </svg>
          <style dangerouslySetInnerHTML={{
            __html: `
            .dash-anim { animation: dash 2s linear infinite; }
            @keyframes dash { to { stroke-dashoffset: 0; } }
          `}} />
        </div>
      )
    },
  ];

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const cards = gsap.utils.toArray('.protocol-card');
      const inners = gsap.utils.toArray('.protocol-card-inner');

      cards.forEach((card, index) => {
        if (index < cards.length - 1) {
          const nextCard = cards[index + 1];
          gsap.to(inners[index], {
            scale: 0.9,
            opacity: 0.5,
            filter: 'blur(10px)',
            ease: 'none',
            scrollTrigger: {
              trigger: nextCard,
              start: 'top bottom',
              end: 'top top',
              scrub: true,
            }
          });
        }
      });
    }, containerRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={containerRef} className="bg-background pt-12 pb-24">
      <div className="text-center mb-12 px-6">
        <h2 className="font-sans font-bold text-4xl uppercase tracking-tighter">The Protocol</h2>
        <p className="font-mono text-sm text-dark/50 mt-4">Methodology for scalable autonomy</p>
      </div>

      <div className="relative">
        {steps.map((step, i) => (
          <div
            key={i}
            className="protocol-card h-screen w-full flex items-center justify-center px-4 sticky top-0"
          >
            <div className="protocol-card-inner relative w-full max-w-4xl h-[60vh] md:h-[70vh] bg-primary rounded-[3rem] border border-dark/10 shadow-2xl overflow-hidden flex flex-col justify-end p-8 md:p-16">

              <step.Anim />

              <div className="relative z-10 flex flex-col md:flex-row md:items-end justify-between gap-8">
                <div>
                  <div className="font-mono text-xl text-accent mb-4">[{step.num}]</div>
                  <h3 className="font-sans font-bold text-4xl md:text-6xl text-dark uppercase tracking-tighter leading-none mb-4">
                    {step.title}
                  </h3>
                  <p className="font-mono text-sm text-dark/70 max-w-md">
                    {step.desc}
                  </p>
                </div>
              </div>

            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ==========================================
// 6. MEMBERSHIP - "Access the Blueprint"
// ==========================================
function Membership() {
  return (
    <section className="bg-primary text-dark py-32 px-6 flex flex-col items-center justify-center text-center relative z-10 w-full overflow-hidden">
      <div className="absolute inset-0 z-0 bg-background/50 rounded-t-[4rem] -mx-4 md:-mx-16" />
      <div className="relative z-10">
        <h2 className="font-sans font-bold text-5xl md:text-7xl uppercase tracking-tighter mb-6">
          Ready to Access?
        </h2>
        <p className="font-mono text-lg text-dark/70 max-w-xl mx-auto mb-12">
          Join the network of operators scaling physical and digital autonomy for Latin America.
        </p>
        <button className="magnetic-btn bg-accent text-primary px-10 py-5 rounded-full font-sans font-bold text-xl overflow-hidden group">
          <span className="magnetic-btn-text flex items-center gap-3 group-hover:text-primary transition-colors">
            Access the Blueprint <ArrowRight size={24} />
          </span>
        </button>
      </div>
    </section>
  );
}

// ==========================================
// 7. FOOTER
// ==========================================
function Footer() {
  return (
    <footer className="bg-dark text-primary rounded-t-[4rem] px-8 py-16 md:px-16 md:py-24 mt-[-2rem] relative z-20">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between gap-12">
        <div className="flex-1">
          <h2 className="font-sans font-bold text-3xl md:text-4xl uppercase tracking-tighter mb-4">Lazarus</h2>
          <p className="font-mono text-sm text-primary/50 max-w-sm mb-8">
            Scaling physical and digital autonomy for the next era of Latin America.
          </p>
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.7)]" />
            <span className="font-mono text-xs text-primary/70 uppercase">System Operational</span>
          </div>
        </div>

        <div className="flex gap-16 font-mono text-sm">
          <div className="flex flex-col gap-4 text-primary/50">
            <a href="#hardware" className="hover:text-accent transition-colors block hover:-translate-y-[1px]">Hardware</a>
            <a href="#digital" className="hover:text-accent transition-colors block hover:-translate-y-[1px]">Digital</a>
            <a href="#bridge" className="hover:text-accent transition-colors block hover:-translate-y-[1px]">Bridge</a>
          </div>
          <div className="flex flex-col gap-4 text-primary/50">
            <a href="#" className="hover:text-accent transition-colors block hover:-translate-y-[1px]">Privacy</a>
            <a href="#" className="hover:text-accent transition-colors block hover:-translate-y-[1px]">Terms</a>
            <a href="#" className="hover:text-accent transition-colors block hover:-translate-y-[1px]">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
}

// ==========================================
// MAIN APP COMPONENT
// ==========================================
function App() {
  return (
    <div className="bg-background min-h-screen text-dark selection:bg-accent selection:text-primary">
      <Navbar />
      <Hero />
      <Features />
      <Philosophy />
      <Protocol />
      <Membership />
      <Footer />
    </div>
  );
}

export default App;
