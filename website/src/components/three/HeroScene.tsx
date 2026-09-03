"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame, type ThreeElements } from "@react-three/fiber";
import { Float } from "@react-three/drei";
import * as THREE from "three";

/* 玻璃质感的中央晶体 — icosahedron + 自定义物理材质光泽 */
function Crystal(props: ThreeElements["mesh"]) {
  const mesh = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!mesh.current) return;
    const t = clock.getElapsedTime();
    mesh.current.rotation.x = Math.sin(t * 0.3) * 0.15;
    mesh.current.rotation.y = t * 0.18;
  });

  return (
    <mesh ref={mesh} {...props}>
      <icosahedronGeometry args={[1.35, 1]} />
      <meshPhysicalMaterial
        color="#8b7dd8"
        metalness={0.1}
        roughness={0.12}
        transmission={0.92}
        thickness={1.6}
        ior={1.45}
        clearcoat={1}
        clearcoatRoughness={0.08}
        iridescence={0.9}
        iridescenceIOR={1.6}
        transparent
      />
    </mesh>
  );
}

/* 环绕粒子场 — 固定随机种子避免水合不一致 */
function Particles({ count = 220 }: { count?: number }) {
  const points = useRef<THREE.Points>(null);

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    let seed = 42;
    const rand = () => {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    };
    for (let i = 0; i < count * 3; i++) {
      arr[i] = (rand() - 0.5) * 9;
    }
    return arr;
  }, [count]);

  useFrame(({ clock }) => {
    if (!points.current) return;
    points.current.rotation.y = clock.getElapsedTime() * 0.04;
    points.current.rotation.z = Math.sin(clock.getElapsedTime() * 0.1) * 0.05;
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.045}
        color="#9ee7ff"
        transparent
        opacity={0.65}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/* 环绕轨道环 — 强化"分析中"的仪器感 */
function OrbitRings() {
  const g = useRef<THREE.Group>(null);
  useFrame(({ pointer }) => {
    if (!g.current) return;
    g.current.rotation.x = 0.5 + pointer.y * 0.15;
    g.current.rotation.y = pointer.x * 0.3;
  });
  return (
    <group ref={g}>
      <mesh rotation={[Math.PI / 2.4, 0, 0]}>
        <torusGeometry args={[2.3, 0.012, 8, 128]} />
        <meshBasicMaterial color="#67e0ff" transparent opacity={0.4} />
      </mesh>
      <mesh rotation={[Math.PI / 1.8, 0.4, 0]}>
        <torusGeometry args={[2.85, 0.008, 8, 128]} />
        <meshBasicMaterial color="#c084fc" transparent opacity={0.3} />
      </mesh>
    </group>
  );
}

export default function HeroScene() {
  return (
    <Canvas
      dpr={[1, 1.8]}
      camera={{ position: [0, 0, 5.2], fov: 45 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      aria-hidden="true"
    >
      <ambientLight intensity={0.35} />
      <pointLight position={[5, 5, 5]} intensity={28} color="#7dd3fc" />
      <pointLight position={[-5, -3, 3]} intensity={20} color="#a78bfa" />
      <Float speed={1.4} rotationIntensity={0.25} floatIntensity={0.7}>
        <Crystal />
      </Float>
      <OrbitRings />
      <Particles />
    </Canvas>
  );
}
