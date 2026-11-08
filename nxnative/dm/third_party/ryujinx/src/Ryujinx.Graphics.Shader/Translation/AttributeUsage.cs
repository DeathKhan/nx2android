using System;
using System.Collections.Generic;
using System.Numerics;

namespace Ryujinx.Graphics.Shader.Translation
{
    class AttributeUsage
    {
        public bool NextUsesFixedFuncAttributes { get; private set; }
        public int UsedInputAttributes { get; private set; }
        public int UsedOutputAttributes { get; private set; }
        public HashSet<int> UsedInputAttributesPerPatch { get; }
        public HashSet<int> UsedOutputAttributesPerPatch { get; }
        public HashSet<int> NextUsedInputAttributesPerPatch { get; private set; }
        public int PassthroughAttributes { get; private set; }
        // nxnative: the fragment inputs Ampere's LDTRAM reads per vertex (InstEmitAmpereMem.ALdtram): declared as arrays of the primitive's three vertices (SPIR-V PerVertexKHR)
        public int PerVertexInputAttributes { get; private set; }
        private int _nextUsedInputAttributes;
        private int _thisUsedInputAttributes;
        private Dictionary<int, int> _perPatchAttributeLocations;
        private readonly IGpuAccessor _gpuAccessor;

        public UInt128 NextInputAttributesComponents { get; private set; }
        public UInt128 ThisInputAttributesComponents { get; private set; }

        public AttributeUsage(IGpuAccessor gpuAccessor)
        {
            _gpuAccessor = gpuAccessor;

            UsedInputAttributesPerPatch = [];
            UsedOutputAttributesPerPatch = [];
        }

        public void SetInputUserAttribute(int index, int component)
        {
            int mask = 1 << index;

            UsedInputAttributes |= mask;
            _thisUsedInputAttributes |= mask;
            ThisInputAttributesComponents |= UInt128.One << (index * 4 + component);
        }

        public void SetInputUserAttributePerVertex(int index)
        {
            PerVertexInputAttributes |= 1 << index;
        }

        public bool IsPerVertexInput(int index) => (PerVertexInputAttributes & (1 << index)) != 0;

        // nxnative: without VK_KHR_fragment_shader_barycentric (IGpuAccessor.QueryHostSupportsFragmentShaderBarycentric) a per-vertex read of input L at vertex v is a read of a
        // flat input: L itself for vertex 0 (the interpolated input is no longer declared there), and for vertices 1 and 2 locations above every input the shader reads,
        // taken in order of first use; a generated geometry shader writes the three from the primitive's vertices (shader_vk.py pv_geometry). The shader's inputs are
        // all known before any instruction is emitted (the decoder runs first), so the first allocation sees the whole mask. Past location 31 (32 user attributes)
        // there is no room, and the shader is refused
        private readonly Dictionary<int, int[]> _perVertexFlatLocations = new();
        private int _perVertexFlatNext = -1;

        public IReadOnlyDictionary<int, int[]> PerVertexFlatLocations => _perVertexFlatLocations;

        public bool IsPerVertexFlatLocation(int location)
        {
            foreach (int[] locations in _perVertexFlatLocations.Values)
            {
                if (Array.IndexOf(locations, location) >= 0)
                {
                    return true;
                }
            }

            return false;
        }

        public int GetPerVertexFlatLocation(int location, int vertex)
        {
            if (!_perVertexFlatLocations.TryGetValue(location, out int[] locations))
            {
                if (_perVertexFlatNext < 0)
                {
                    _perVertexFlatNext = UsedInputAttributes == 0 ? 0 : 32 - BitOperations.LeadingZeroCount((uint)UsedInputAttributes);
                }

                locations = new int[3];
                locations[0] = location;

                for (int v = 1; v < 3; v++)
                {
                    if (_perVertexFlatNext > 31)
                    {
                        throw new InvalidOperationException($"no input location left for the per-vertex read of attribute {location} at vertex {v} (32 in use)");
                    }

                    locations[v] = _perVertexFlatNext++;
                }

                _perVertexFlatLocations.Add(location, locations);
            }

            return locations[vertex];
        }

        public void SetInputUserAttributePerPatch(int index)
        {
            UsedInputAttributesPerPatch.Add(index);
        }

        public void SetOutputUserAttribute(int index)
        {
            UsedOutputAttributes |= 1 << index;
        }

        public void SetOutputUserAttributePerPatch(int index)
        {
            UsedOutputAttributesPerPatch.Add(index);
        }

        public void MergeFromtNextStage(bool gpPassthrough, bool nextUsesFixedFunctionAttributes, AttributeUsage nextStage)
        {
            NextInputAttributesComponents = nextStage.ThisInputAttributesComponents;
            NextUsedInputAttributesPerPatch = nextStage.UsedInputAttributesPerPatch;
            NextUsesFixedFuncAttributes = nextUsesFixedFunctionAttributes;
            MergeOutputUserAttributes(gpPassthrough, nextStage.UsedInputAttributes, nextStage.UsedInputAttributesPerPatch);

            if (UsedOutputAttributesPerPatch.Count != 0)
            {
                // Regular and per-patch input/output locations can't overlap,
                // so we must assign on our location using unused regular input/output locations.

                Dictionary<int, int> locationsMap = new();

                int freeMask = ~UsedOutputAttributes;

                foreach (int attr in UsedOutputAttributesPerPatch)
                {
                    int location = BitOperations.TrailingZeroCount(freeMask);
                    if (location == 32)
                    {
                        _gpuAccessor.Log($"No enough free locations for patch input/output 0x{attr:X}.");
                        break;
                    }

                    locationsMap.Add(attr, location);
                    freeMask &= ~(1 << location);
                }

                // Both stages must agree on the locations, so use the same "map" for both.
                _perPatchAttributeLocations = locationsMap;
                nextStage._perPatchAttributeLocations = locationsMap;
            }
        }

        private void MergeOutputUserAttributes(bool gpPassthrough, int mask, IEnumerable<int> perPatch)
        {
            _nextUsedInputAttributes = mask;

            if (gpPassthrough)
            {
                PassthroughAttributes = mask & ~UsedOutputAttributes;
            }
            else
            {
                UsedOutputAttributes |= mask;
                UsedOutputAttributesPerPatch.UnionWith(perPatch);
            }
        }

        public int GetPerPatchAttributeLocation(int index)
        {
            if (_perPatchAttributeLocations == null || !_perPatchAttributeLocations.TryGetValue(index, out int location))
            {
                return index;
            }

            return location;
        }

        public bool IsUsedOutputAttribute(int attr)
        {
            // The check for fixed function attributes on the next stage is conservative,
            // returning false if the output is just not used by the next stage is also valid.
            if (NextUsesFixedFuncAttributes &&
                attr >= AttributeConsts.UserAttributeBase &&
                attr < AttributeConsts.UserAttributeEnd)
            {
                int index = (attr - AttributeConsts.UserAttributeBase) >> 4;
                return (_nextUsedInputAttributes & (1 << index)) != 0;
            }

            return true;
        }

        public int GetFreeUserAttribute(bool isOutput, int index)
        {
            int useMask = isOutput ? _nextUsedInputAttributes : _thisUsedInputAttributes;
            int bit = -1;

            while (useMask != -1)
            {
                bit = BitOperations.TrailingZeroCount(~useMask);

                if (bit == 32)
                {
                    bit = -1;
                    break;
                }
                else if (index < 1)
                {
                    break;
                }

                useMask |= 1 << bit;
                index--;
            }

            return bit;
        }

        public void SetAllInputUserAttributes()
        {
            UsedInputAttributes |= Constants.AllAttributesMask;
            ThisInputAttributesComponents |= ~UInt128.Zero >> (128 - Constants.MaxAttributes * 4);
        }

        // nxnative: the attributes a passthrough geometry shader forwards when the next stage is not known (TranslatorContext.Translate)
        public void SetPassthroughWithoutNextStage(int mask)
        {
            _nextUsedInputAttributes = mask;
            PassthroughAttributes = mask & ~UsedOutputAttributes;
        }

        public void SetAllOutputUserAttributes()
        {
            UsedOutputAttributes |= Constants.AllAttributesMask;
        }
    }
}
