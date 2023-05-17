using System;
using System.Diagnostics;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

class GraphChart : MonoBehaviour
{
    [BurstCompile]
    struct UE3DVecJob : IJob
    {
        public NativeArray<Vector3> A;
        public NativeArray<Vector3> B;
        public NativeArray<Vector3> C;

        public void Execute()
        {
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }

    [BurstCompile]
    struct UM3DVecJob : IJob
    {
        public NativeArray<float3> A;
        public NativeArray<float3> B;
        public NativeArray<float3> C;

        public void Execute()
        {
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }

    [BurstCompile]
    struct UEMat4Job : IJob
    {
        public NativeArray<Matrix4x4> A;
        public NativeArray<Matrix4x4> B;
        public NativeArray<Matrix4x4> C;

        public void Execute()
        {
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = A[i] * B[i];
            }
        }
    }

    [BurstCompile]
    struct UMMat4Job : IJob
    {
        public NativeArray<float4x4> A;
        public NativeArray<float4x4> B;
        public NativeArray<float4x4> C;

        public void Execute()
        {
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = A[i] * B[i];
            }
        }
    }

    [BurstCompile]
    struct UEQuatJob : IJob
    {
        public NativeArray<Quaternion> A;
        public NativeArray<Quaternion> B;
        public NativeArray<Quaternion> C;

        public void Execute()
        {
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = A[i] * B[i];
            }
        }
    }

    [BurstCompile]
    struct UMQuatJob : IJob
    {
        public NativeArray<quaternion> A;
        public NativeArray<quaternion> B;
        public NativeArray<quaternion> C;

        public void Execute()
        {
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = math.mul(A[i], B[i]);
            }
        }
    }

    void Start()
    {
        const int size = 1000000;
        const Allocator alloc = Allocator.TempJob;
        NativeArray<Vector3> v3a = new NativeArray<Vector3>(size, alloc);
        NativeArray<Vector3> v3b = new NativeArray<Vector3>(size, alloc);
        NativeArray<Vector3> v3c = new NativeArray<Vector3>(size, alloc);
        NativeArray<float3> f3a = new NativeArray<float3>(size, alloc);
        NativeArray<float3> f3b = new NativeArray<float3>(size, alloc);
        NativeArray<float3> f3c = new NativeArray<float3>(size, alloc);
        NativeArray<Matrix4x4> m4a = new NativeArray<Matrix4x4>(size, alloc);
        NativeArray<Matrix4x4> m4b = new NativeArray<Matrix4x4>(size, alloc);
        NativeArray<Matrix4x4> m4c = new NativeArray<Matrix4x4>(size, alloc);
        NativeArray<float4x4> f4a = new NativeArray<float4x4>(size, alloc);
        NativeArray<float4x4> f4b = new NativeArray<float4x4>(size, alloc);
        NativeArray<float4x4> f4c = new NativeArray<float4x4>(size, alloc);
        NativeArray<Quaternion> ueqa = new NativeArray<Quaternion>(size, alloc);
        NativeArray<Quaternion> ueqb = new NativeArray<Quaternion>(size, alloc);
        NativeArray<Quaternion> ueqc = new NativeArray<Quaternion>(size, alloc);
        NativeArray<quaternion> umqa = new NativeArray<quaternion>(size, alloc);
        NativeArray<quaternion> umqb = new NativeArray<quaternion>(size, alloc);
        NativeArray<quaternion> umqc = new NativeArray<quaternion>(size, alloc);
        for (int i = 0; i < size; ++i)
        {
            v3a[i] = Vector3.zero;
            v3b[i] = Vector3.zero;
            v3c[i] = Vector3.zero;
            f3a[i] = float3.zero;
            f3b[i] = float3.zero;
            f3c[i] = float3.zero;
            m4a[i] = Matrix4x4.identity;
            m4b[i] = Matrix4x4.identity;
            m4c[i] = Matrix4x4.identity;
            f4a[i] = float4x4.identity;
            f4b[i] = float4x4.identity;
            f4c[i] = float4x4.identity;
            ueqa[i] = Quaternion.identity;
            ueqb[i] = Quaternion.identity;
            ueqc[i] = Quaternion.identity;
            umqa[i] = quaternion.identity;
            umqb[i] = quaternion.identity;
            umqc[i] = quaternion.identity;
        }
        UE3DVecJob v3j = new UE3DVecJob { A = v3a, B = v3b, C = v3c };
        UM3DVecJob f3j = new UM3DVecJob { A = f3a, B = f3b, C = f3c };
        UEMat4Job m4j = new UEMat4Job { A = m4a, B = m4b, C = m4c };
        UMMat4Job f4j = new UMMat4Job { A = f4a, B = f4b, C = f4c };
        UEQuatJob ueqj = new UEQuatJob { A = ueqa, B = ueqb, C = ueqc };
        UMQuatJob umqj = new UMQuatJob { A = umqa, B = umqb, C = umqc };

        const int reps = 100;
        long[] v3t = new long[reps];
        long[] f3t = new long[reps];
        long[] m4t = new long[reps];
        long[] f4t = new long[reps];
        long[] ueqt = new long[reps];
        long[] umqt = new long[reps];
        Stopwatch sw = new Stopwatch();
        for (int i = 0; i < reps; ++i)
        {
            sw.Restart();
            v3j.Run();
            v3t[i] = sw.ElapsedTicks;

            sw.Restart();
            f3j.Run();
            f3t[i] = sw.ElapsedTicks;

            sw.Restart();
            m4j.Run();
            m4t[i] = sw.ElapsedTicks;

            sw.Restart();
            f4j.Run();
            f4t[i] = sw.ElapsedTicks;

            sw.Restart();
            ueqj.Run();
            ueqt[i] = sw.ElapsedTicks;

            sw.Restart();
            umqj.Run();
            umqt[i] = sw.ElapsedTicks;
        }

        print(
            "Job,UnityEngine Time,Unity.Mathematics Timen" +
            "3D Vector," + Median(v3t) + "," + Median(f3t) + "n" +
            "4x4 Matrix," + Median(m4t) + "," + Median(f4t) + "n" +
            "Quaternion," + Median(ueqt) + "," + Median(umqt));

        v3a.Dispose();
        v3b.Dispose();
        v3c.Dispose();
        f3a.Dispose();
        f3b.Dispose();
        f3c.Dispose();
        m4a.Dispose();
        m4b.Dispose();
        m4c.Dispose();
        f4a.Dispose();
        f4b.Dispose();
        f4c.Dispose();
        ueqa.Dispose();
        ueqb.Dispose();
        ueqc.Dispose();
        umqa.Dispose();
        umqb.Dispose();
        umqc.Dispose();

        Application.Quit();
    }

    static long Median(long[] values)
    {
        Array.Sort(values);
        return values[values.Length / 2];
    }
}