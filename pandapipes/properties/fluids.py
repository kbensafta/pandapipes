# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import itertools
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import CoolProp
from CoolProp.CoolProp import PropsSI
from chemicals.identifiers import IDs_to_CASs, MW
from chemicals.critical import Tc, Pc, Zc
from chemicals import dipole_moment
from chemicals.vectorized.viscosity import Wilke
from chemicals.vectorized import Lucas_gas, lemmon2000_rho, lemmon2000_air_MW

from pandapipes import pp_dir
from pandapipes.properties.properties_toolbox import calculate_mixture_density, calculate_mixture_viscosity, \
    calculate_mixture_molar_mass, calculate_molar_fraction_from_mass_fraction, calculate_mixture_heat_capacity, \
    calculate_mixture_compressibility, calculate_mixture_calorific_values, calculate_mass_fraction_from_molar_fraction
from pandapipes.constants import P_CONVERSION, NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapower.io_utils import JSONSerializableClass

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Fluid(JSONSerializableClass):
    """

    """

    def __init__(self, name, fluid_type, **kwargs):
        """

        :param name:
        :type name:
        :param fluid_type:
        :type fluid_type:
        :param kwargs:
        :type kwargs:
        """
        super(Fluid, self).__init__()
        self.name = name
        if not isinstance(fluid_type, str) or fluid_type.lower() not in ["gas", "liquid"]:
            logger.warning("The fluid %s has the fluid type %s which might cause problems in the "
                           "pipeflow calculation, as it expects either 'gas' or 'liquid'."
                           % (name, fluid_type))
        self.fluid_type = fluid_type.lower()
        self.is_gas = self.fluid_type == "gas"
        self.all_properties = kwargs
        for prop_name, prop in self.all_properties.items():
            if not isinstance(prop, FluidProperty):
                logger.warning("The property %s was not defined as a fluid property. This might "
                               "cause problems when trying to ask for values." % prop_name)

    def __repr__(self):
        """
        Definition of fluid representation in the console.

        :return: representation of fluid in the console
        :rtype: str
        """

        r = "Fluid %s (%s) with properties:" % (self.name, self.fluid_type)
        for key in self.all_properties.keys():
            r += "\n   - %s (%s)" % (key, self.all_properties[key].__class__.__name__[13:])
        return r

    def add_property(self, property_name, prop, overwrite=True, warn_on_duplicates=True):
        """
        This function adds a new property.

        :param property_name: Name of the new property
        :type property_name: str
        :param prop: Values for the property, for example a curve or just a constant value
        :type prop: pandapipes.FluidProperty
        :param overwrite: True if existing property with the same name shall be overwritten
        :type overwrite: bool
        :param warn_on_duplicates: True, if a warning of properties with the same name should be
                                    returned
        :type warn_on_duplicates: bool

        :Example:
            >>> fluid.add_property('water_density', pandapipes.FluidPropertyConstant(998.2061),\
                                   overwrite=True, warn_on_duplicates=False)

        """
        if property_name in self.all_properties:
            if warn_on_duplicates:
                ow_string = "It will be overwritten." if overwrite else "It will not be replaced."
                logger.warning("The property %s already exists. %s" % (property_name, ow_string))
            if not overwrite:
                return
        self.all_properties[property_name] = prop

    def get_property(self, property_name, *at_values):
        """
        This function returns the value of the requested property.

        :param property_name: Name of the searched property
        :type property_name: str
        :param at_values: Value for which the property should be returned
        :type at_values:
        :return: Returns property at the certain value
        :rtype: pandapipes.FluidProperty
        """

        if property_name not in self.all_properties:
            raise UserWarning("The property %s was not defined for the fluid %s"
                              % (property_name, self.name))
        return self.all_properties[property_name].get_at_value(*at_values)

    def get_density(self, temperature=NORMAL_TEMPERATURE, pressure=NORMAL_PRESSURE, phase='gas', mode='quality'):
        """
        This function returns the density at a certain temperature.

        :param temperature: Temperature at which the density is queried
        :type temperature: float
        :return: Density at the required temperature

        """
        if phase == 'gas':
            if self.name == 'hgas':
                density = 0.8047119641901379
            elif self.name == 'lgas':
                density = 0.8261278456699505
            else:
                if mode == 'performance':
                    method = 'SRK'
                elif mode == 'quality':
                    method = 'HEOS'
                else:
                    logger.error("Mode is ambigious. Please choose 'quality' or 'performance' ")
                fluid = CoolProp.AbstractState('HEOS', self.name)
                fluid.specify_phase(CoolProp.iphase_gas)
                fluid.update(CoolProp.PT_INPUTS, NORMAL_PRESSURE, NORMAL_TEMPERATURE)
                density = fluid.rhomass()
        elif phase == 'liquid':
            if mode == 'performance':
                if self.name == 'water':
                    fluid = CoolProp.AbstractState('IF97', 'Water')
                    fluid.update(CoolProp.PT_INPUTS, pressure * P_CONVERSION, temperature)
                    density = fluid.rhomass()
                else:
                    fluid = CoolProp.AbstractState("INCOMP", self.name)
                    fluid.update(CoolProp.PT_INPUTS, pressure * P_CONVERSION, temperature)
                    density = fluid.rhomass()
            elif mode == 'quality':
                fluid = CoolProp.AbstractState('HEOS', self.name)
                fluid.specify_phase(CoolProp.iphase_liquid)
                fluid.update(CoolProp.PT_INPUTS, pressure * P_CONVERSION, temperature)
                density = fluid.rhomass()
        else:
            logger.error("Phase is undefined. Please check your input")
        return density

    def get_viscosity(self, temperature, p_bar=NORMAL_PRESSURE, phase='gas', mode='performance'):
        """
        This function returns the viscosity at a certain temperature.

        :param temperature: Temperature at which the viscosity is queried
        :type temperature: float
        :return: Viscosity at the required temperature

        """
        if not mode == 'quality' and not mode == 'performance':
            logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

        if phase == 'gas':
            if self.name == 'hgas':
                CAS = ['74-82-8', '7727-37-9', '124-38-9', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3']
                mole_fractions = [0.8957, 0.0159, 0.0077, 0.0615, 0.0125, 0.0048, 0.0009, 0.001]
                molar_mass_list = [16.043, 28.0134, 44.0095, 30.07, 44.097, 58.124, 72.151, 86.178]
                t_crit = [190.564, 126.2, 304.2, 305.32, 369.83, 425.12, 469.7, 507.6]
                p_crit = [4599000.0, 3394387.5, 7376460.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3025000.0]
                z_crit = [0.286, 0.29, 0.274, 0.279, 0.277, 0.274, 0.268, 0.264]
                dipole_moment_dbye = [0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0]
                if mode == 'performance':
                    viscosity_list = [Lucas_gas(temperature, t_crit[i], p_crit[i], z_crit[i], molar_mass_list[i], dipole_moment_dbye[i]) for i in range(len(CAS))]
                    dyn_viscosity = Wilke(mole_fractions, viscosity_list, molar_mass_list)
                elif mode == 'quality':
                    viscosity_hgas = CoolProp.AbstractState("HEOS",
                                                            'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
                    viscosity_hgas.set_mole_fractions(mole_fractions)
                    viscosity_hgas.specify_phase(CoolProp.iphase_gas)
                    viscosity_hgas.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                    dyn_viscosity = viscosity_hgas.viscosity()
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

            elif self.name == 'lgas':
                CAS = ['74-82-8', '7727-37-9', '124-38-9', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3']
                mole_fractions = [0.8433, 0.0981, 0.0145, 0.0341, 0.006, 0.0023, 0.001, 0.0007]
                molar_mass_list = [16.043, 28.0134, 44.0095, 30.07, 44.097, 58.124, 72.151, 86.178]
                t_crit = [190.564, 126.2, 304.2, 305.32, 369.83, 425.12, 469.7, 507.6]
                p_crit = [4599000.0, 3394387.5, 7376460.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3025000.0]
                z_crit = [0.286, 0.29, 0.274, 0.279, 0.277, 0.274, 0.268, 0.264]
                dipole_moment_dbye = [0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0]
                if mode == 'performance':
                    viscosity_list = [Lucas_gas(temperature, t_crit[i], p_crit[i], z_crit[i],
                                                molar_mass_list[i], dipole_moment_dbye[i]) for i in range(len(CAS))]
                    dyn_viscosity = Wilke(mole_fractions, viscosity_list, molar_mass_list)
                elif mode == 'quality':
                    viscosity_lgas = CoolProp.AbstractState("HEOS",
                                                            'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
                    viscosity_lgas.set_mole_fractions(mole_fractions)
                    viscosity_lgas.specify_phase(CoolProp.iphase_gas)
                    viscosity_lgas.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                    dyn_viscosity = viscosity_lgas.viscosity()
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
            elif self.name == 'carbondioxide':
                CAS = '124-38-9'
                t_crit = 304.2
                p_crit = 7376460.0
                z_crit = 0.274
                mole_mass = 44.0095
                dipole_moment_dbye = 0.0
                if mode == 'performance':
                    dyn_viscosity = Lucas_gas(temperature, t_crit, p_crit, z_crit, mole_mass, dipole_moment_dbye)
                elif mode == 'quality':
                    dyn_viscosity = PropsSI("V", "P", p_bar * P_CONVERSION, "T", temperature, CAS)
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
            elif self.name == 'air':
                viscosity_air = CoolProp.AbstractState('HEOS', "Air")
                viscosity_air.specify_phase(CoolProp.iphase_gas)
                viscosity_air.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                dyn_viscosity = viscosity_air.viscosity()
            else:
                CAS = IDs_to_CASs(self.name)[0]
                if mode == 'performance':
                    dyn_viscosity = Lucas_gas(temperature, Tc(CAS), Pc(CAS), Zc(CAS), MW(CAS), dipole_moment(CAS))
                elif mode == 'quality':
                    dyn_viscosity = PropsSI("V", "P", p_bar * P_CONVERSION, "T", temperature, CAS)
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
        elif phase == 'liquid':
            if self.name == 'water':
                if mode == 'performance':
                    viscosity_water = CoolProp.AbstractState("IF97", "Water")
                    viscosity_water.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                    dyn_viscosity = viscosity_water.viscosity()
                elif mode == 'quality':
                    viscosity_water = CoolProp.AbstractState("HEOS", "Water")
                    viscosity_water.specify_phase(CoolProp.iphase_liquid)
                    viscosity_water.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                    dyn_viscosity = viscosity_water.viscosity()
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
            else:
                if mode == 'performance':
                    viscosity_liquid = CoolProp.AbstractState("INCOMP", self.name)
                    viscosity_liquid.specify_phase(CoolProp.iphase_liquid)
                    viscosity_liquid.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                    dyn_viscosity = viscosity_liquid.viscosity()
                elif mode == 'quality':
                    viscosity_liquid = CoolProp.AbstractState("HEOS", self.name)
                    viscosity_liquid.specify_phase(CoolProp.iphase_liquid)
                    viscosity_liquid.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature)
                    dyn_viscosity = viscosity_liquid.viscosity()
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
        return dyn_viscosity

    def get_heat_capacity(self, temperature, p_bar=1.01325, mode="performance", phase="gas"):
        """
        This function returns the heat capacity at a certain temperature.

        :param temperature: Temperature at which the heat capacity is queried
        :type temperature: float
        :return: Heat capacity at the required temperature

        """
        if phase == 'gas':
            if self.name == 'hgas':
                if mode == 'performance':

                    heat_capacity_hgas = CoolProp.AbstractState("SRK", 'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
                elif mode == 'quality':
                    heat_capacity_hgas = CoolProp.AbstractState("HEOS",
                                                                'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

                heat_capacity_array = np.empty(len(temperature), dtype=np.float64)
                heat_capacity_hgas.set_mole_fractions([0.8957, 0.0159, 0.0077, 0.0615, 0.0125, 0.0048, 0.0009, 0.001])
                heat_capacity_hgas.specify_phase(CoolProp.iphase_gas)
                for i in range(len(temperature)):
                    heat_capacity_hgas.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature[i])
                    heat_capacity_array[i] = heat_capacity_hgas.cpmass()
                heat_capacity = heat_capacity_array.copy()
            elif self.name == 'lgas':
                if mode == 'performance':
                    heat_capacity_lgas = CoolProp.AbstractState("SRK",
                                                                'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
                elif mode == 'quality':
                    heat_capacity_lgas = CoolProp.AbstractState("HEOS",
                                                                'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
                heat_capacity_array = np.empty(len(temperature), dtype=np.float64)
                heat_capacity_lgas.set_mole_fractions([0.8433, 0.0981, 0.0145, 0.0341, 0.006, 0.0023, 0.001, 0.0007])
                heat_capacity_lgas.specify_phase(CoolProp.iphase_gas)
                for i in range(len(temperature)):
                    heat_capacity_lgas.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature[i])
                    heat_capacity_array[i] = heat_capacity_lgas.cpmass()
                heat_capacity = heat_capacity_array.copy()
            else:
                if mode == 'performance':
                    heat_capacity_AS = CoolProp.AbstractState('SRK', self.name)
                elif mode == 'quality':
                    heat_capacity_AS = CoolProp.AbstractState('HEOS', self.name)
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
                heat_capacity_array = np.empty([len(temperature)], dtype=np.float64)
                heat_capacity_AS.specify_phase(CoolProp.iphase_gas)
                for i in range(len(temperature)):
                    heat_capacity_AS.update(CoolProp.PT_INPUTS, p_bar * P_CONVERSION, temperature[i])
                    heat_capacity_array[i] = heat_capacity_AS.cpmass()
                heat_capacity = heat_capacity_array.copy()
        elif phase == 'liquid':
            if self.name == 'water':
                if mode == 'performance':
                    heat_capacity_water = CoolProp.AbstractState("IF97", "Water")
                elif mode == 'quality':
                    heat_capacity_water = CoolProp.AbstractState("HEOS", "Water")
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
                heat_capacity_array = np.empty(len(temperature), dtype=np.float64)
                heat_capacity_water.specify_phase(CoolProp.iphase_liquid)
                for i in range(len(temperature)):
                    heat_capacity_water.update(CoolProp.PT_INPUTS, p_bar[i] * P_CONVERSION, temperature[i])
                    heat_capacity_array[i] = heat_capacity_water.cpmass()
                heat_capacity = heat_capacity_water.cpmass()
            else:
                if mode == 'performance':
                    heat_capacity_liquid = CoolProp.AbstractState("INCOMP", self.name)
                elif mode == 'quality':
                    heat_capacity_liquid = CoolProp.AbstractState("HEOS", self.name)
                else:
                    logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")
                heat_capacity_array = np.empty(len(temperature), dtype=np.float64)
                heat_capacity_liquid.specify_phase(CoolProp.iphase_liquid)
                for i in range(len(temperature)):
                    heat_capacity_liquid.update(CoolProp.PT_INPUTS, p_bar[i] * P_CONVERSION, temperature[i])
                    heat_capacity_array[i] = heat_capacity_liquid.cpmass()
                heat_capacity = heat_capacity_liquid.cpmass()
        return heat_capacity

    def get_molar_mass(self):
        """
        This function returns the molar mass.

        :return: molar mass

        """
        if self.name == 'hgas':
            molar_mass = 17.984362784
        elif self.name == 'lgas':
            molar_mass = 18.471238304
        elif self.name == 'carbondioxide':
            molar_mass = 44.0098
        elif self.name == 'air':
            molar_mass = 28.96546
        else:
            molar_mass = MW(self.name)
        return molar_mass

    def get_compressibility(self, p_bar, temperature = np.array([285.15]), mode='quality'):
        """
        This function returns the compressibility at a certain pressure.

        :param p_bar: pressure at which the compressibility is queried
        :type p_bar: float or array of floats
        :return: compressibility at the required pressure

        """
        if self.name == 'hgas':
            compressibility_hgas_norm = 0.9970936857549283
            if mode == 'performance':
                compressibility_hgas = CoolProp.AbstractState("SRK",
                                                              'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
            elif mode == 'quality':
                compressibility_hgas = CoolProp.AbstractState("HEOS",
                                                              'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
            else:
                logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

            compressibility_hgas_array = np.empty(len(p_bar), dtype=np.float64)
            compressibility_hgas.set_mole_fractions([0.8957, 0.0159, 0.0077, 0.0615, 0.0125, 0.0048, 0.0009, 0.001])
            compressibility_hgas.specify_phase(CoolProp.iphase_gas)
            for i in range(len(p_bar)):
                compressibility_hgas.update(CoolProp.PT_INPUTS, p_bar[i], temperature)
                compressibility_hgas_array[i] = compressibility_hgas.compressibility_factor()
            compressibility_factor_hgas = compressibility_hgas_array.copy()
            gas_law_deviation_coefficient = compressibility_factor_hgas / compressibility_hgas_norm
        elif self.name == 'lgas':
            compressibility_lgas_norm = 0.9975395425980074
            if mode == 'performance':
                compressibility_lgas = CoolProp.AbstractState("SRK",
                                                              'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
            elif mode == 'quality':
                compressibility_lgas = CoolProp.AbstractState("HEOS",
                                                              'Methane&Nitrogen&CarbonDioxide&Ethane&Propane&Butane&Pentane&Hexane')
            else:
                logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

            compressibility_lgas_array = np.empty(len(p_bar), dtype=np.float64)
            compressibility_lgas.set_mole_fractions([0.8957, 0.0159, 0.0077, 0.0615, 0.0125, 0.0048, 0.0009, 0.001])
            compressibility_lgas.specify_phase(CoolProp.iphase_gas)
            for i in range(len(p_bar)):
                compressibility_lgas.update(CoolProp.PT_INPUTS, p_bar[i], temperature)
                compressibility_lgas_array[i] = compressibility_lgas.compressibility_factor()
            compressibility_factor_lgas = compressibility_lgas_array.copy()
            gas_law_deviation_coefficient = compressibility_factor_lgas / compressibility_lgas_norm
        else:
            if mode == 'performance':
                compressibility_gas = CoolProp.AbstractState('SRK', self.name)
                compressibility_gas_norm = CoolProp.AbstractState('SRK', self.name)
            elif mode == 'quality':
                compressibility_gas = CoolProp.AbstractState('HEOS', self.name)
                compressibility_gas_norm = CoolProp.AbstractState('HEOS', self.name)
            else:
                logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

            compressibility_gas_array = np.empty(len(p_bar), dtype=np.float64)
            compressibility_gas.specify_phase(CoolProp.iphase_gas)
            compressibility_gas_norm.specify_phase(CoolProp.iphase_gas)
            for i in range(len(p_bar)):
                compressibility_gas.update(CoolProp.PT_INPUTS, p_bar[i], temperature)
                compressibility_gas_array[i] = compressibility_gas.compressibility_factor()
            compressibility_gas_norm.update(CoolProp.PT_INPUTS, NORMAL_PRESSURE, NORMAL_TEMPERATURE)
            compressibility_factor_gas = compressibility_gas_array.copy()
            compressibility_factor_gas_norm = compressibility_gas_norm.compressibility_factor()
            gas_law_deviation_coefficient = compressibility_factor_gas / compressibility_factor_gas_norm
        return gas_law_deviation_coefficient

    def get_der_compressibility(self, p_bar=np.array([1.01325])):
        """
        This function returns the derivative of the compressibility with respect to pressure.

        :return: derivative of the compressibility

        """
        #der_gldc = (self.get_compressibility(p_bar * P_CONVERSION) - 1) / p_bar
        der_gldc = -0.022
        return der_gldc

    def get_lower_heating_value(self):
        return self.get_property("lhv")

    def get_higher_heating_value(self):
        return self.get_property("hhv")


class FluidProperty(JSONSerializableClass):
    """
    Property Base Class
    """

    def __init__(self):
        """

        """
        super().__init__()

    def get_at_value(self, *args):
        """

        :param args:
        :type args:
        :return:
        :rtype: float, np.array
        """
        raise NotImplementedError("Please implement a proper fluid property!")

    def get_at_integral_value(self, *args):
        """

        :param args:
        :type args:
        :return:
        :rtype:
        """
        raise NotImplementedError("Please implement a proper fluid property!")


class FluidPropertyInterExtra(FluidProperty):
    """
    Creates Property with interpolated or extrapolated values.
    """
    json_excludes = JSONSerializableClass.json_excludes + ["prop_getter"]
    prop_getter_entries = {"x": "x", "y": "y", "_fill_value_orig": "fill_value"}

    def __init__(self, x_values, y_values, method="interpolate_extrapolate"):
        """

        :param x_values:
        :type x_values:
        :param y_values:
        :type y_values:
        :param method:
        :type method:
        """
        super(FluidPropertyInterExtra, self).__init__()
        if method.lower() == "interpolate_extrapolate":
            self.prop_getter = interp1d(x_values, y_values, fill_value="extrapolate")
        else:
            self.prop_getter = interp1d(x_values, y_values)

    def get_at_value(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the y-values \
            of the property are to be displayed
        :type arg: str, float or array
        :return: y-value/s
        :rtype: float, array
        """
        return self.prop_getter(arg)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        mean = (self.prop_getter(upper_limit_arg) + self.prop_getter(upper_limit_arg)) / 2
        return mean * (upper_limit_arg - lower_limit_arg)

    @classmethod
    def from_path(cls, path, method="interpolate_extrapolate"):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path: Target path of the txt file
        :type path: str
        :param method: Method with which the values are to be interpolated
        :type method: str
        :return: interpolated values
        :rtype: pandapipes.FluidProperty
        """
        values = np.loadtxt(path)
        return cls(values[:, 0], values[:, 1], method=method)

    def to_dict(self):
        d = super(FluidPropertyInterExtra, self).to_dict()
        d.update({k: self.prop_getter.__dict__[k] for k in self.prop_getter_entries.keys()})
        # d.update({"x_values": self.prop_getter.x, "y_values": self.prop_getter.y,
        #           "method": "interpolate_extrapolate"
        #           if self.prop_getter.fill_value == "extrapolate" else None})
        return d

    @classmethod
    def from_dict(cls, d):
        obj = JSONSerializableClass.__new__(cls)
        d2 = {cls.prop_getter_entries[k]: v for k, v in d.items()
              if k in cls.prop_getter_entries.keys()}
        d3 = {k: v for k, v in d.items() if k not in cls.prop_getter_entries.keys()}
        d3["prop_getter"] = interp1d(**d2)
        obj.__dict__.update(d3)
        return obj


class FluidPropertyConstant(FluidProperty):
    """
    Creates Property with a constant value.
    """

    def __init__(self, value, warn_dependent_variables=False):
        """

        :param value:
        :type value:
        """
        super(FluidPropertyConstant, self).__init__()
        self.value = value
        self.warn_dependent_variables = warn_dependent_variables

    def get_at_value(self, *args):
        """

        :param args: Name of the property
        :type args: str
        :return: Value of the property
        :rtype: float

        :Example:
            >>> heat_capacity = get_fluid(net).all_properties["heat_capacity"].get_at_value(293.15)
        """
        if len(args) > 1:
            raise UserWarning('Please define either none or an array-like argument')
        elif len(args) == 1:
            if self.warn_dependent_variables:
                logger.warning('Constant property received several input variables, although it is'
                               'independent of these')
            output = np.array([self.value]) * np.ones(len(args[0]))
        else:
            output = np.array([self.value])
        return output

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        if isinstance(upper_limit_arg, pd.Series):
            ul = self.value * upper_limit_arg.values
        else:
            ul = self.value * np.array(upper_limit_arg)
        if isinstance(lower_limit_arg, pd.Series):
            ll = self.value * lower_limit_arg.values
        else:
            ll = self.value * np.array(lower_limit_arg)
        return ul - ll

    @classmethod
    def from_path(cls, path):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path:
        :type path:
        :return:
        :rtype:
        """
        value = np.loadtxt(path).item()
        return cls(value)

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        if "warn_dependent_variables" not in obj.__dict__.keys():
            obj.__dict__["warn_dependent_variables"] = False
        return obj


class FluidPropertyLinear(FluidProperty):
    """
    Creates Property with a linear course.
    """

    def __init__(self, slope, offset):
        """

        :param slope:
        :type slope:
        :param offset:
        :type offset:

        """
        super(FluidPropertyLinear, self).__init__()
        self.slope = slope
        self.offset = offset

    def get_at_value(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the function \
            of the property should be calculated
        :type arg: str, float or array
        :return: y-value or function values
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["compressibility"].get_at_value(p_bar)

        """
        if isinstance(arg, pd.Series):
            return self.offset + self.slope * arg.values
        else:
            return self.offset + self.slope * np.array(arg)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        if isinstance(upper_limit_arg, pd.Series):
            ul = self.offset * upper_limit_arg.values + 0.5 * self.slope * np.power(
                upper_limit_arg.values, 2)
        else:
            ul = self.offset * np.array(upper_limit_arg) + 0.5 * self.slope * np.array(
                np.power(upper_limit_arg.values, 2))
        if isinstance(lower_limit_arg, pd.Series):
            ll = self.offset * lower_limit_arg.values + 0.5 * self.slope * np.power(
                lower_limit_arg.values, 2)
        else:
            ll = self.offset * np.array(lower_limit_arg) + 0.5 * self.slope * np.array(
                np.power(lower_limit_arg.values, 2))
        return ul - ll

    @classmethod
    def from_path(cls, path):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path:
        :type path:
        :return:
        :rtype:
        """
        slope, offset = np.loadtxt(path)
        return cls(slope, offset)


class FluidPropertyPolynominal(FluidProperty):
    """
    Creates Property with a polynominal course.
    """

    def __init__(self, x_values, y_values, polynominal_degree):
        """

        :param x_values:
        :type x_values:
        :param y_values:
        :type y_values:
        :param polynominal_degree:
        :type polynominal_degree:
        """
        super(FluidPropertyPolynominal, self).__init__()
        const = np.polyfit(x_values, y_values, polynominal_degree)
        self.prop_getter = np.poly1d(const)
        self.prop_int_getter = np.polyint(self.prop_getter)

    def get_at_value(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the function \
            of the property should be calculated
        :type arg: float or list-like objects
        :return: y-value or function values
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_value(t_k)

        """
        return self.prop_getter(arg)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        return self.prop_int_getter(upper_limit_arg) - self.prop_int_getter(lower_limit_arg)

    @classmethod
    def from_path(cls, path, polynominal_degree):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path: Target path of the txt file
        :type path: str
        :param polynominal_degree: degree of the polynominal
        :type polynominal_degree: int
        :return: Fluid object
        :rtype: pandapipes.FluidProperty
        """
        values = np.loadtxt(path)
        return cls(values[:, 0], values[:, 1], polynominal_degree)


def create_constant_property(net, fluid_name, property_name, value, overwrite=True, warn_on_duplicates=True):
    """
    Creates a property with a constant value.

    :param net: Name of the network to which the property is added
    :type net: pandapipesNet
    :param property_name: Name of the new property
    :type property_name: str
    :param value: Constant value of the property
    :type value: float
    :param overwrite:  True if existing property with the same name shall be overwritten
    :type overwrite: basestring
    :param warn_on_duplicates: True, if a warning of properties with the same name should be
                                returned
    :type warn_on_duplicates: basestring
    """
    prop = FluidPropertyConstant(value)
    net.fluid[fluid_name].add_property(property_name, prop, overwrite=overwrite,
                                       warn_on_duplicates=warn_on_duplicates)
    return prop


def create_linear_property(net, fluid_name, property_name, slope, offset, overwrite=True,
                           warn_on_duplicates=True):
    """
    Creates a property with a linear correlation.

    :param net: Name of the network to which the property is added
    :type net: pandapipesNet
    :param property_name: Name of the new property
    :type property_name: str
    :param slope: Slope of the linear correlation
    :type slope: float
    :param offset: Offset of the linear function
    :type offset: float
    :param overwrite:  True if existing property with the same name shall be overwritten
    :type overwrite: basestring
    :param warn_on_duplicates: True, if a warning of properties with the same name should be
                                returned
    :type warn_on_duplicates: basestring
    """
    prop = FluidPropertyLinear(slope, offset)
    net.fluid[fluid_name].add_property(property_name, prop, overwrite=overwrite,
                                       warn_on_duplicates=warn_on_duplicates)
    return prop


def create_constant_fluid(name=None, fluid_type=None, **kwargs):
    """
    Creates a constant fluid.

    :param name: Name of the fluid
    :type name: str
    :param fluid_type: Type of the fluid
    :type fluid_type: str
    :param kwargs: Additional information
    :return: Fluid
    :rtype: Fluid
    """
    properties = dict()
    for prop_name, prop in kwargs.items():
        properties[str(prop_name)] = FluidPropertyConstant(prop)

    return Fluid(name=name, fluid_type=fluid_type, **properties)


def call_lib(fluid_name):
    """
    Creates a fluid with default fluid properties.

    :param fluid_name: Fluid which should be used
    :type fluid_name: str
    :return: Fluid - Chosen fluid with default fluid properties
    :rtype: Fluid
    """

    def interextra_property(prop):
        return FluidPropertyInterExtra.from_path(
            os.path.join(pp_dir, "properties", fluid_name, prop + ".txt"))

    def constant_property(prop):
        return FluidPropertyConstant.from_path(
            os.path.join(pp_dir, "properties", fluid_name, prop + ".txt"))

    def linear_property(prop):
        return FluidPropertyLinear.from_path(
            os.path.join(pp_dir, "properties", fluid_name, prop + ".txt"))

    liquids = ["water"]
    gases = ["air", "lgas", "hgas", "hydrogen", "methane", "ethane", "butane", "propane", "carbondioxide", "nitrogen"]

    if fluid_name == "natural_gas":
        logger.error("'natural_gas' is ambigious. Please choose 'hgas' or 'lgas' "
                     "(high- or low calorific natural gas)")
    if fluid_name not in liquids and fluid_name not in gases:
        raise AttributeError("Fluid '%s' not found in the fluid library. It might not be "
                             "implemented yet." % fluid_name)

    phase = "liquid" if fluid_name in liquids else "gas"

    density = interextra_property("density")
    viscosity = interextra_property("viscosity")
    heat_capacity = interextra_property("heat_capacity")
    molar_mass = constant_property("molar_mass")
    der_compr = constant_property("der_compressibility")
    compr = linear_property("compressibility")

    if (phase == 'gas'):
        lhv = constant_property("lower_heating_value")
        hhv = constant_property("higher_heating_value")

        return Fluid(fluid_name, phase, density=density, viscosity=viscosity,
                     heat_capacity=heat_capacity, molar_mass=molar_mass,
                     compressibility=compr, der_compressibility=der_compr, lhv=lhv, hhv=hhv)
    else:
        return Fluid(fluid_name, phase, density=density, viscosity=viscosity,
                     heat_capacity=heat_capacity, molar_mass=molar_mass, compressibility=compr,
                     der_compressibility=der_compr)


def get_fluid(net, fluid_name):
    """
    This function shows which fluid is used in the net.

    :param net: Current network
    :type net: pandapipesNet
    :return: Fluid - Name of the fluid which is used in the current network
    :rtype: Fluid
    """
    if fluid_name not in net.fluid.keys():
        raise AttributeError("There is no fluid defined for the given net!")
    fluid = net.fluid[fluid_name]
    if not isinstance(fluid, Fluid):
        logger.warning("The fluid in this net is not of the pandapipes Fluid type. This could lead"
                       " to errors, as some components might depend on this structure")
    return fluid


def _add_fluid_to_net(net, fluid, overwrite=True):
    """
    Adds a fluid to a net. If overwrite is False, a warning is printed and the fluid is not set.

    :param net: The pandapipes network for which to set fluid
    :type net: pandapipesNet
    :param fluid: fluid which to insert into the network
    :type fluid: Fluid
    :param overwrite: If True, an existing fluid will just be overwritten, otherwise a warning is\
        printed out and the fluid is not reset.
    :type overwrite: bool, default True
    :return: No output.
    :type: None
    """
    if "fluid" in net and fluid.name in net["fluid"] and not overwrite:
        logger.warning("The fluid %s would replace the exisiting fluid with the same name and thus cannot be created. "
                       "Try to set overwrite to True" % (fluid.name))
        return

    if isinstance(fluid, str):
        logger.warning("Instead of a pandapipes.Fluid, a string ('%s') was passed to the fluid "
                       "argument. Internally, it will be passed to call_lib(fluid) to get the "
                       "respective pandapipes.Fluid." % fluid)
        fluid = call_lib(fluid)
    net["fluid"][fluid.name] = fluid


def get_property(net, property_name, fluid_name=None, *at_values):
    if len(net._fluid) == 1:
        return get_fluid(net, net._fluid[0]).get_property(property_name, *at_values)
    else:
        return net.fluid[fluid_name].get_property(property_name, *at_values)

def get_mixture_molar_mass(net, mass_fraction):
    fluid_list = net._fluid.copy()
    MW_list = np.empty(len(fluid_list), dtype=np.float64)
    for i in range(len(fluid_list)):
        if fluid_list[i] == 'hgas':
            MW_list[i] = 17.984362784
        elif fluid_list[i] == 'lgas':
            MW_list[i] = 18.471238304
        elif fluid_list[i] == 'carbondioxide':
            MW_list[i] = 44.0098
        else:
            MW_list[i] = MW(fluid_list[i])
    return calculate_mixture_molar_mass(MW_list, component_proportions=mass_fraction)


def get_mixture_density(net, temperature, mass_fraction, mode="quality"):
    print(np.shape(mass_fraction))
    density_list = np.empty(len(mass_fraction), dtype=np.float64)
    fluid_list = '&'.join(net._fluid)
    if mode == 'performance':
        density_mixture = CoolProp.AbstractState('SRK', fluid_list)
        for i in range(len(mass_fraction)):
            density_mixture.set_mass_fractions(mass_fraction[i])
            density_mixture.update(CoolProp.PT_INPUTS, NORMAL_PRESSURE * P_CONVERSION, NORMAL_TEMPERATURE)
            density_list[i] = density_mixture.rhomass()
    elif mode == 'quality':
        density_mixture = CoolProp.AbstractState('HEOS', fluid_list)
        for i in range(len(mass_fraction)):
            density_mixture.set_mass_fractions(mass_fraction[i])
            density_mixture.update(CoolProp.PT_INPUTS, NORMAL_PRESSURE * P_CONVERSION, NORMAL_TEMPERATURE)
            density_list[i] = density_mixture.rhomass()
    else:
        logger.error('Mode is not available. Check your input.')
    return density_list

def get_mixture_viscosity(net, temperature, mass_fraction, pressure=np.array([NORMAL_PRESSURE]), mode='performance'):
    if not mode == 'quality' and not mode == 'performance':
        logger.error("Mode is not valid. Please choose 'quality' or 'performance' ")

    fluid_list = net._fluid.copy()
    viscosity_list = np.empty(len(fluid_list), dtype=np.float64)
    molar_weight_list = np.empty(len(fluid_list), dtype=np.float64)
    for i in range(len(fluid_list)):
        if fluid_list[i] == 'hgas':
            CAS = ['74-82-8', '7727-37-9', '124-38-9', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3']
            mole_fractions = [0.8957, 0.0159, 0.0077, 0.0615, 0.0125, 0.0048, 0.0009, 0.001]
            molar_mass_list = [16.043, 28.0134, 44.0095, 30.07, 44.097, 58.124, 72.151, 86.178]
            t_crit = [190.564, 126.2, 304.2, 305.32, 369.83, 425.12, 469.7, 507.6]
            p_crit = [4599000.0, 3394387.5, 7376460.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3025000.0]
            z_crit = [0.286, 0.29, 0.274, 0.279, 0.277, 0.274, 0.268, 0.264]
            dipole_moment_dbye = [0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0]
            if mode == 'performance':
                viscosity = [Lucas_gas(temperature, t_crit[i], p_crit[i], z_crit[i],
                                       molar_mass_list[i], dipole_moment_dbye[i]) for i in range(len(CAS))]
            else:
                viscosity = [PropsSI("V", "P", 1.01325 * P_CONVERSION, "T", temperature, fluid) for fluid in CAS]
            viscosity_list[i] = Wilke(mole_fractions, viscosity, molar_mass_list)
            molar_weight_list[i] = 17.984362784
        elif fluid_list[i] == 'lgas':
            CAS = ['74-82-8', '7727-37-9', '124-38-9', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3']
            mole_fractions = [0.8433, 0.0981, 0.0145, 0.0341, 0.006, 0.0023, 0.001, 0.0007]
            molar_mass_list = [16.043, 28.0134, 44.0095, 30.07, 44.097, 58.124, 72.151, 86.178]
            t_crit = [190.564, 126.2, 304.2, 305.32, 369.83, 425.12, 469.7, 507.6]
            p_crit = [4599000.0, 3394387.5, 7376460.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3025000.0]
            z_crit = [0.286, 0.29, 0.274, 0.279, 0.277, 0.274, 0.268, 0.264]
            dipole_moment_dbye = [0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0]
            if mode == 'performance':
                viscosity = [Lucas_gas(temperature, t_crit[i], p_crit[i], z_crit[i],
                                       molar_mass_list[i], dipole_moment_dbye[i]) for i in range(len(CAS))]
            else:
                viscosity = [PropsSI("V", "P", 1.01325 * P_CONVERSION, "T", temperature, fluid) for fluid in CAS]
            viscosity_list[i] = Wilke(mole_fractions, viscosity, molar_mass_list)
            molar_weight_list[i] = 18.471238304
        elif fluid_list[i] == 'carbondioxide':
            CAS = '124-38-9'
            t_crit = 304.2
            p_crit = 7376460.0
            z_crit = 0.274
            mole_mass = 44.0095
            dipole_moment_dbye = 0.0
            if mode == 'performance':
                viscosity_list[i] = PropsSI("CPMASS", "P", pressure * P_CONVERSION, "T", temperature, "CO2")
            else:
                viscosity_list[i] = PropsSI("V", "P", NORMAL_PRESSURE * P_CONVERSION, "T", temperature, CAS)
            molar_weight_list[i] = 44.0095
        else:
            CAS = IDs_to_CASs(fluid_list[i])[0]
            if mode == 'performance':
                t_crit = Tc(CAS)
                p_crit = Pc(CAS)
                z_crit = Zc(CAS)
                mole_mass = MW(CAS)
                dipole_moment_dbye = dipole_moment(CAS)
                p = np.full(np.shape(temperature), 101325., dtype=np.float64)
                viscosity_list[i] = PropsSI("V", "P", 101325., "T", 273.15, CAS)
            else:
                viscosity_list[i] = PropsSI("V", "P", NORMAL_PRESSURE * P_CONVERSION, "T", temperature, CAS)
            molar_weight_list[i] = MW(fluid_list[i])
    molar_fraction = calculate_molar_fraction_from_mass_fraction(mass_fraction.T, molar_weight_list)
    viscosity = Wilke(molar_fraction, viscosity_list, molar_weight_list)
    return viscosity

def get_mixture_heat_capacity(net, temperature, mass_fraction):
    heat_capacity_list = [net.fluid[fluid].get_heat_capacity(temperature) for fluid in net._fluid]
    return calculate_mixture_heat_capacity(heat_capacity_list, mass_fraction.T)

def get_mixture_compressibility(net, pressure, mass_fraction, temperature=NORMAL_TEMPERATURE, calc_mode="HEOS"):
    compressibility_list = np.empty(len(mass_fraction), dtype=np.float64)
    norm_compressibility_list = np.empty(len(mass_fraction), dtype=np.float64)
    fluid_list = '&'.join(net._fluid)
    compressibility_mixture = CoolProp.AbstractState(calc_mode, fluid_list)
    norm_compressibility_mixture = CoolProp.AbstractState(calc_mode, fluid_list)
    for i in range(len(mass_fraction)):
        compressibility_mixture.set_mass_fractions(mass_fraction[i])
        norm_compressibility_mixture.set_mass_fractions(mass_fraction[i])
        compressibility_mixture.update(CoolProp.PT_INPUTS, pressure[i] * P_CONVERSION, temperature)
        norm_compressibility_mixture.update(CoolProp.PT_INPUTS, NORMAL_PRESSURE, NORMAL_TEMPERATURE)
        compressibility_list[i] = compressibility_mixture.compressibility_factor()
        norm_compressibility_list[i] = norm_compressibility_mixture.compressibility_factor()
    gldc_list = compressibility_list / norm_compressibility_list
    return gldc_list

def get_mixture_der_cmpressibility(net, pressure, mass_fraction, temperature=NORMAL_TEMPERATURE, calc_mode="HEOS"):
    fluid_list = [net._fluid[i] for i in range(len(net._fluid))]
    compressibility_list = np.empty(len(pressure))
    compressibility_list_norm = np.empty(len(pressure))
    fluid_list = [i.title() for i in fluid_list]
    fluid_list = '&'.join(fluid_list)
    compressibility_mixture = CoolProp.AbstractState(calc_mode, fluid_list)
    compressibility_mixture_norm = CoolProp.AbstractState(calc_mode, fluid_list)
    for i in range(len(pressure)):
        compressibility_mixture.set_mass_fractions(mass_fraction[i])
        compressibility_mixture.update(CoolProp.PT_INPUTS, pressure[i], temperature)
        compressibility_mixture_norm.set_mass_fractions(mass_fraction[i])
        compressibility_mixture_norm.update(CoolProp.PT_INPUTS, NORMAL_PRESSURE, NORMAL_TEMPERATURE)
        compressibility_list[i] = compressibility_mixture.compressibility_factor()
        compressibility_list_norm[i] = compressibility_mixture.compressibility_factor()
    gldc_param_list = compressibility_list / compressibility_list_norm - 1
    return gldc_param_list


def get_mixture_higher_heating_value(net, mass_fraction):
    calorific_list = np.array([net.fluid[fluid].get_property('hhv') for fluid in net._fluid])
    return calculate_mixture_calorific_values(calorific_list, mass_fraction.T)


def get_mixture_lower_heating_value(net, mass_fraction):
    calorific_list = np.array([net.fluid[fluid].get_property('lhv') for fluid in net._fluid])
    return calculate_mixture_calorific_values(calorific_list, mass_fraction.T)


def is_fluid_gas(net):
    if len(net._fluid) == 1:
        return get_fluid(net, net._fluid[0]).is_gas
    else:
        state = [get_fluid(net, fluid).is_gas for fluid in net._fluid]
        if np.all(state):
            return True
        elif np.all(~np.array(state)):
            return False
        else:
            logger.warning('Be careful. You look at system containing both fluid and gaseous fluids.')


def create_individual_fluid(fluid_name, fluid_components,
                            temperature_list, pressure_list,
                            component_proportions, proportion_type='mass', phase='gas'):
    molar_mass = []
    density = []
    viscosity = []
    heat_capacity = []
    compressibility = []
    der_compressibility = []
    high_calorific = []
    low_calorific = []
    for fl_co in fluid_components:
        fluid = call_lib(fl_co)
        molar_mass += [fluid.get_molar_mass()]
        density += [fluid.get_density()]
        viscosity += [fluid.get_viscosity(temperature_list)]
        heat_capacity += [fluid.get_heat_capacity(temperature_list)]
        compressibility += [fluid.get_property('compressibility', pressure_list)]
        der_compressibility += [fluid.get_property('der_compressibility', temperature_list)]
        high_calorific += [fluid.get_property('hhv')]
        low_calorific += [fluid.get_property('lhv')]
    if proportion_type == 'mass':
        mof = calculate_molar_fraction_from_mass_fraction(component_proportions, molar_mass)
        maf = np.array(component_proportions)
    elif proportion_type == 'molar':
        mof = np.array(component_proportions)
        maf = calculate_mass_fraction_from_molar_fraction(component_proportions, molar_mass)
    else:
        raise (AttributeError('proportion type %s not defined. Select either mass or molar' %proportion_type))
    dens = calculate_mixture_density(density, maf)
    visc = calculate_mixture_viscosity(viscosity, mof, np.array(molar_mass))
    heat = calculate_mixture_heat_capacity(heat_capacity, maf)
    comp = calculate_mixture_compressibility(compressibility, maf)
    derc = calculate_mixture_compressibility(der_compressibility, maf)
    mass = calculate_mixture_molar_mass(molar_mass, maf)
    higc = calculate_mixture_calorific_values(np.array(high_calorific), maf)
    lowc = calculate_mixture_calorific_values(np.array(low_calorific), maf)

    dens = FluidPropertyInterExtra(temperature_list, dens)
    visc = FluidPropertyInterExtra(temperature_list, visc)
    heat = FluidPropertyInterExtra(temperature_list, heat)
    mass = FluidPropertyConstant(mass)
    higc = FluidPropertyConstant(higc)
    lowc = FluidPropertyConstant(lowc)
    derc = FluidPropertyInterExtra(temperature_list, derc)
    comp = FluidPropertyInterExtra(pressure_list, comp)

    fluid = Fluid(fluid_name, phase, density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                  der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    return fluid
